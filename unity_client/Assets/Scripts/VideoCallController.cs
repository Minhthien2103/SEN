using UnityEngine;
using UnityEngine.UI; // Dùng để can thiệp đổi màu nút

public class VideoCallController:MonoBehaviour
{
    private bool isMicOn = true;
    private bool isCameraOn = false;

    [Header("Giao diện Nút")]
    public Image micButtonImage; // Nơi kéo cái nút Mic vào để đổi màu
    public Color micOnColor = Color.green;
    public Color micOffColor = Color.red;

    public void ToggleMic()
    {
        isMicOn = !isMicOn; // Đảo trạng thái

        // Đổi màu nút để người dùng biết
        if (micButtonImage != null)
        {
            micButtonImage.color = isMicOn ? micOnColor : micOffColor;
        }
        
        Debug.Log("Trạng thái Mic hiện tại: " + (isMicOn ? "BẬT" : "TẮT"));
        // GHI CHÚ: Sau này sẽ nối API ghi âm giọng nói người dùng vào đây
    }

    public void ToggleCamera()
    {
        isCameraOn = !isCameraOn;
        Debug.Log("Chế độ Camera AI quét vật thể: " + (isCameraOn ? "BẬT" : "TẮT"));
        
        // GHI CHÚ ĐỒ ÁN: Chỗ này là nơi gọi API sang Python để bật Webcam, 
        // chạy YOLO hoặc model nhận diện vật phẩm, rồi đẩy kết quả vào Tủ đồ!
    }

    public void EndCall()
    {
        Debug.Log("Đã cúp máy. Đóng ứng dụng!");
        
        // Tắt ứng dụng khi đã Build ra file .exe
        Application.Quit();
        
        // Tắt chế độ Play khi đang test trong Unity Editor
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #endif
    }
}