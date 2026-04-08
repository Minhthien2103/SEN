"""
FILE: ObjectGenerationBackend.py
MÔ TẢ:
    Hệ thống Ảo hóa Đồ vật Vật lý thành Tài sản Game (Game Asset).
    Pipeline 2 bước:
    1. VLM (Gemini Vision): Phân tích hình ảnh camera thực tế để trích xuất prompt miêu tả đồ vật.
    2. Image Generation (FLUX/Stable Diffusion): Chuyển đổi prompt thành hình ảnh 2D phong cách anime/cel-shaded.
    Tích hợp cơ chế đa luồng (multi-threading) để tạo trước các biến thể trạng thái (hư hỏng, cũ) chạy nền,
    không làm nghẽn luồng kết nối của Unity Client.
"""

import io
import json
import time
import threading
import requests
import google.generativeai as genai
from PIL import Image

# Import Console từ rich để đồng bộ màu sắc log với các module khác
try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("Lỗi: Hãy chạy lệnh 'pip install rich'")

class ObjectGenerationBackend:
    """
    Quản lý luồng giao tiếp với các API GenAI (Gemini & Hugging Face)
    để tạo ra hình ảnh vật phẩm 2D cho hệ thống Inventory của nhân vật.
    """

    def __init__(self, gemini_api_key: str, hf_api_key: str):
        """
        Khởi tạo và cấu hình các client API.

        Args:
            gemini_api_key (str): Khóa API để truy cập dịch vụ Google Gemini (VLM).
            hf_api_key (str): Khóa API để gọi các Serverless Inference API của Hugging Face.
        """
        console.print("[cyan]Đang khởi tạo Hệ thống Tạo sinh Đồ vật (VLM + Diffusion)...[/cyan]")
        genai.configure(api_key=gemini_api_key)
        
        self.hf_token = hf_api_key
        
        # Chọn phiên bản Flash để tối ưu tốc độ nhận diện ảnh (low-latency) thay vì bản Pro
        self.vlm = genai.GenerativeModel('gemini-2.5-flash')
        
        # Hậu tố ép phong cách: Đảm bảo hình ảnh sinh ra luôn có nền trắng (dễ dàng tách nền sau này) 
        # và thống nhất một phong cách đồ họa Anime 3D.
        self.ART_STYLE_SUFFIX = ", 3D game asset prop, stylized anime, cel-shaded, vibrant colors, solid white background, highly detailed"
        
        # Prompt mồi (System Prompt) ép Gemini chỉ trả về duy nhất 1 câu tiếng Anh miêu tả vật thể chính,
        # bỏ qua hoàn toàn các yếu tố gây nhiễu như bàn tay người cầm hoặc phông nền lộn xộn.
        self.vlm_prompt = """
        You are a prompt engineer for an AI game asset generator.
        Describe the SINGLE main object in this image in ONE concise English sentence.
        Focus on color, material, and shape. Ignore the background, hands, or UI elements.
        Example: "A sleek silver laptop with a glowing blue logo."
        """

    def _call_real_image_api(self, prompt: str, filename_to_save: str) -> str:
        """
        Gửi yêu cầu render ảnh tới Serverless Endpoint của Hugging Face.
        Xử lý cơ chế Cold-Start (khi model bị đưa vào trạng thái ngủ do lâu không dùng).

        Args:
            prompt (str): Câu lệnh miêu tả hình ảnh bằng tiếng Anh.
            filename_to_save (str): Tên file lưu trữ ảnh đầu ra trên ổ đĩa nội bộ.

        Returns:
            str: Trả về đường dẫn ảo 'local_path:/...' nếu thành công, hoặc chuỗi báo lỗi.
        """
        console.print(f"\n[cyan]Đang gọi Hugging Face vẽ:[/cyan] '{prompt}'")
        
        # Sử dụng FLUX.1-schnell vì tốc độ sinh ảnh dưới 2 giây/tấm, lý tưởng cho tính năng real-time.
        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                with open(filename_to_save, 'wb') as f:
                    f.write(response.content)
                console.print(f"[green]Đã lưu siêu phẩm AI vào file: {filename_to_save}[/green]")
                return f"local_path:/{filename_to_save}" 
            
            # Xử lý Cold Start: Model Serverless bị hạ xuống (scale down) để tiết kiệm tài nguyên.
            # Đọc thời gian ước tính từ server trả về, cho tiến trình ngủ (sleep) rồi thử lại.
            elif response.status_code == 503:
                err_msg = response.json()
                wait_time = err_msg.get('estimated_time', 20)
                console.print(f"[yellow]Máy chủ Hugging Face đang khởi động mô hình. Cần đợi khoảng {round(wait_time)} giây...[/yellow]")
                time.sleep(wait_time)
                console.print("[cyan]Thử gọi lại lần 2...[/cyan]")
                
                retry_resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                if retry_resp.status_code == 200:
                    with open(filename_to_save, 'wb') as f:
                        f.write(retry_resp.content)
                    console.print(f"[green]Đã lưu file sau khi đợi: {filename_to_save}[/green]")
                    return f"local_path:/{filename_to_save}"
                else:
                    raise Exception(f"Lỗi sau khi retry: {retry_resp.text}")
            else:
                raise Exception(f"Lỗi Server (Code {response.status_code}): {response.text}")
                
        except Exception as e:
            console.print(f"[red]Không thể tạo ảnh: {e}[/red]")
            return "error_generation"

    def _generate_variations_background(self, base_item_name: str, base_prompt: str):
        """
        Hàm chạy trên luồng nền (Background Thread).
        Tự động tạo ra phiên bản cũ/hư hỏng của vật phẩm vừa thu thập được.
        Mục đích: Chuẩn bị sẵn dữ liệu (Pre-compute) để LLM có thể sử dụng sau này 
        (ví dụ: kịch bản vật phẩm bị hao mòn theo thời gian) mà không bắt người chơi phải chờ load lại.
        """
        console.print("\n" + "-"*30)
        console.print(f"[cyan][BACKGROUND TASK] Đang âm thầm tạo biến thể cho:[/cyan] {base_item_name}...")
        
        variation_prompt = f"{base_prompt}, slightly damaged, worn out, old" + self.ART_STYLE_SUFFIX
        # Giới hạn tên file ngắn để tránh lỗi độ dài hệ thống file của Windows
        filename = f"item_{base_item_name[:5].replace(' ', '_')}_variant.jpg"
        
        self._call_real_image_api(variation_prompt, filename_to_save=filename)
        console.print(f"[green][BACKGROUND TASK] Đã cất biến thể '{filename}' vào Tủ Ký Ức.[/green]")
        console.print("-" * 30)

    def process_scan_request(self, byte_array) -> str:
        """
        Nhận luồng byte hình ảnh từ client Unity, chuyển đổi thành vật phẩm và trả về Payload JSON.

        Args:
            byte_array (bytes): Dữ liệu hình ảnh được chụp từ camera người dùng.

        Returns:
            str: Chuỗi JSON định dạng chuẩn để Unity Client tiến hành hiển thị UI.
        """
        console.print("\n" + "="*50)
        console.print("[cyan][HỆ THỐNG] NHẬN TÍN HIỆU QUÉT ĐỒ VẬT TỪ UNITY![/cyan]")
        start_time = time.time()
        
        try:
            console.print("\n[yellow][1/3] Gemini đang nhìn ảnh...[/yellow]")
            # BytesIO giúp PIL xử lý ảnh trực tiếp trên RAM mà không cần phải ghi file ra đĩa
            img = Image.open(io.BytesIO(byte_array))
            vlm_response = self.vlm.generate_content([self.vlm_prompt, img])
            base_description = vlm_response.text.strip().replace('"', '')
            console.print(f" -> Gemini thấy: {base_description}")
            
            final_prompt = base_description + self.ART_STYLE_SUFFIX
            
            console.print("\n[yellow][2/3] Bắt đầu kích hoạt Cỗ máy vẽ ảnh Hugging Face...[/yellow]")
            main_filename = f"item_main_generated.jpg"
            generated_img_url = self._call_real_image_api(final_prompt, main_filename)
            
            console.print("\n[yellow][3/3] Kích hoạt tiến trình tạo biến thể chạy ngầm...[/yellow]")
            bg_thread = threading.Thread(
                target=self._generate_variations_background, 
                args=(base_description, base_description),
                daemon=True # Đảm bảo thread này chết khi Server tắt
            )
            bg_thread.start()

            process_time = round(time.time() - start_time, 2)
            
            # Payload tuân thủ chuẩn Giao thức WebSocket API đã định nghĩa với Client Unity
            payload = {
                "action": "close_portal_and_show_item",
                "status": "success",
                "item_name": base_description,
                "asset_url": generated_img_url,
                "message": f"Tạo thành công vật phẩm trong {process_time} giây."
            }
            
            console.print(f"\n[green][HOÀN TẤT] Tổng thời gian: {process_time}s[/green]")
            return json.dumps(payload, indent=2, ensure_ascii=False)

        except Exception as e:
            console.print(f"[red]Lỗi xử lý luồng tạo vật phẩm: {e}[/red]")
            error_payload = {
                "action": "close_portal_with_error",
                "status": "error",
                "message": str(e)
            }
            return json.dumps(error_payload, indent=2, ensure_ascii=False)