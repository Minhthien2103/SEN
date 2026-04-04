import io
import json
import time
import threading
import requests
import google.generativeai as genai
from PIL import Image

class ObjectGenerationBackend:
    def __init__(self, gemini_api_key: str, hf_api_key: str):
        print("[S.E.N.S.E] Đang khởi tạo Hệ thống Tạo sinh Đồ vật...")
        genai.configure(api_key=gemini_api_key)
        
        # Token của Hugging Face
        self.hf_token = hf_api_key
        
        self.vlm = genai.GenerativeModel('gemini-2.5-flash')
        
        self.ART_STYLE_SUFFIX = ", 3D game asset prop, stylized anime, cel-shaded, vibrant colors, solid white background, highly detailed"
        
        self.vlm_prompt = """
        You are a prompt engineer for an AI game asset generator.
        Describe the SINGLE main object in this image in ONE concise English sentence.
        Focus on color, material, and shape. Ignore the background, hands, or UI elements.
        Example: "A sleek silver laptop with a glowing blue logo."
        """

    def _call_real_image_api(self, prompt: str, filename_to_save: str) -> str:
        """
        Gọi API Tạo ảnh từ Hugging Face (Mô hình Stable Diffusion XL)
        """
        print(f"\n[API TẠO ẢNH] Đang gọi Hugging Face vẽ: '{prompt}'")
        
        # Gọi mô hình FLUX.1-schnell (Siêu nhanh, siêu đẹp và luôn bật 24/7)
        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}
        
        try:
            # Gửi yêu cầu lên server Hugging Face
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                with open(filename_to_save, 'wb') as f:
                    f.write(response.content)
                print(f"[THÀNH CÔNG] Đã lưu siêu phẩm AI vào file: {filename_to_save} 🎉")
                return f"local_path:/{filename_to_save}" 
            
            # Xử lý trường hợp "Cold Start" (Server đang tải mô hình lên RAM)
            elif response.status_code == 503:
                err_msg = response.json()
                wait_time = err_msg.get('estimated_time', 20)
                print(f"[CẢNH BÁO] Máy chủ Hugging Face đang khởi động mô hình. Cần đợi khoảng {round(wait_time)} giây...")
                time.sleep(wait_time) # Tạm dừng và thử gọi lại 1 lần nữa
                print("[API TẠO ẢNH] Thử gọi lại lần 2...")
                retry_resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                if retry_resp.status_code == 200:
                    with open(filename_to_save, 'wb') as f:
                        f.write(retry_resp.content)
                    print(f"[THÀNH CÔNG] Đã lưu file sau khi đợi: {filename_to_save} 🎉")
                    return f"local_path:/{filename_to_save}"
                else:
                    raise Exception(f"Lỗi sau khi retry: {retry_resp.text}")
            else:
                raise Exception(f"Lỗi Server (Code {response.status_code}): {response.text}")
                
        except Exception as e:
            print(f"[LỖI] Không thể tạo ảnh: {e}")
            return "error_generation"

    def _generate_variations_background(self, base_item_name: str, base_prompt: str):
        print("\n" + "-"*30)
        print(f"[BACKGROUND TASK] Đang âm thầm tạo biến thể cho: {base_item_name}...")
        variation_prompt = f"{base_prompt}, slightly damaged, worn out, old" + self.ART_STYLE_SUFFIX
        filename = f"item_{base_item_name[:5].replace(' ', '_')}_variant.jpg"
        self._call_real_image_api(variation_prompt, filename_to_save=filename)
        print(f"[BACKGROUND TASK] Đã cất biến thể '{filename}' vào Tủ Ký Ức.")
        print("-" * 30)

    def process_scan_request(self, byte_array) -> str:
        print("\n" + "="*50)
        print("[HỆ THỐNG] NHẬN TÍN HIỆU QUÉT ĐỒ VẬT TỪ UNITY!")
        start_time = time.time()
        
        try:
            print("\n[1/3] Gemini đang nhìn ảnh...")
            img = Image.open(io.BytesIO(byte_array))
            vlm_response = self.vlm.generate_content([self.vlm_prompt, img])
            base_description = vlm_response.text.strip().replace('"', '')
            print(f" -> Gemini thấy: {base_description}")
            
            final_prompt = base_description + self.ART_STYLE_SUFFIX
            print("\n[2/3] Bắt đầu kích hoạt Cỗ máy vẽ ảnh Hugging Face...")
            main_filename = f"item_main_generated.jpg"
            generated_img_url = self._call_real_image_api(final_prompt, main_filename)
            
            print("\n[3/3] Kích hoạt tiến trình tạo biến thể chạy ngầm...")
            bg_thread = threading.Thread(
                target=self._generate_variations_background, 
                args=(base_description, base_description)
            )
            bg_thread.start()

            process_time = round(time.time() - start_time, 2)
            payload = {
                "action": "close_portal_and_show_item",
                "status": "success",
                "item_name": base_description,
                "asset_url": generated_img_url,
                "message": f"Tạo thành công vật phẩm trong {process_time} giây."
            }
            
            print(f"\n[HOÀN TẤT] Tổng thời gian: {process_time}s")
            return json.dumps(payload, indent=2, ensure_ascii=False)

        except Exception as e:
            error_payload = {
                "action": "close_portal_with_error",
                "status": "error",
                "message": str(e)
            }
            return json.dumps(error_payload, indent=2, ensure_ascii=False)