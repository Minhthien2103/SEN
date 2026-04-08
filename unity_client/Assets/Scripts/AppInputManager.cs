using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro; 

public class AppInputManager : MonoBehaviour
{
    [Header("Giao diện Camera (Góc phải dưới)")]
    public RawImage cameraDisplay; 
    public bool autoStartCamera = true;
    public float captureInterval = 3f; 
    private WebCamTexture webCamTexture;
    private Texture2D snap; 

    [Header("Giao diện Chat")]
    public static AppInputManager Instance;
    public TextMeshProUGUI chatLogText; 
    public ScrollRect chatScrollRect;   
    public GameObject chatPanel;
    private ConcurrentQueue<Action> mainThreadActions = new ConcurrentQueue<Action>();

    [Header("Cụm UI Ẩn/Hiện (Phải)")]
    public GameObject actionMenuPanel; 
    public RectTransform toggleMenuButton; 
    public float toggleSize = 410; 

    [Header("Cụm UI Ẩn/Hiện (Trái)")]
    public GameObject leftMenuPanel; // CỤM NÚT TRÁI (Chứa nút Chat, Đồ vật 2D)

    [Header("Giao diện Micro")]
    public Image micIcon; 
    public bool isRecording = false;
    public Image micButtonBackground; // Kéo thả component Image của nút Mic vào đây
    public Color micOffColor = Color.red;
    public Color micOnColor = Color.green;

    [Header("Giao diện Quét Đồ Vật 2D")]
    public GameObject objectScanPanel; // Cái khung UI chứa camera đồ vật
    public RawImage objectCameraDisplay; // Cái màn hình hiển thị camera đồ vật

    private string currentDeviceMic;
    private AudioClip recordingClip;
    private float micBlinkSpeed = 5f;

    // ==========================================
    // CÁC HÀM TOGGLE ẨN/HIỆN MENU
    // ==========================================
    public void ToggleLeftMenu()
    {
        if (leftMenuPanel != null) leftMenuPanel.SetActive(!leftMenuPanel.activeSelf);
    }

    public void ToggleActionMenu()
    {
        if (actionMenuPanel != null) actionMenuPanel.SetActive(!actionMenuPanel.activeSelf);
    }

    public void ToggleChat()
    {
        if (chatPanel != null) chatPanel.SetActive(!chatPanel.activeSelf); 
    }

    public void Toggle2DObjectMode()
    {
        if (objectScanPanel != null) 
        {
            bool isActive = !objectScanPanel.activeSelf;
            objectScanPanel.SetActive(isActive);
            
            string status = isActive ? "BẬT" : "TẮT";
            Debug.Log($"🎒 Đã {status} chế độ quét đồ vật 2D!");
        }
    }

    void Start()
    {
        if (Microphone.devices.Length > 0)
        {
            currentDeviceMic = Microphone.devices[0];
        }

        if (WebCamTexture.devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(WebCamTexture.devices[0].name, 640, 480, 30);
            
            if (cameraDisplay != null) 
            {
                cameraDisplay.texture = webCamTexture;
                cameraDisplay.uvRect = new Rect(1, 0, -1, 1); 
            }

            // Gắn camera cho ô Quét Đồ Vật 2D (nếu có)
            if (objectCameraDisplay != null)
            {
                objectCameraDisplay.texture = webCamTexture;
                objectCameraDisplay.uvRect = new Rect(1, 0, -1, 1); 
            }
            
            // Lấy RectTransform của cả cụm Menu Phải để chuẩn bị dời vị trí
            RectTransform menuRect = actionMenuPanel != null ? actionMenuPanel.GetComponent<RectTransform>() : null;

            if (autoStartCamera)
            {
                webCamTexture.Play();
                cameraDisplay.gameObject.SetActive(true);
                // Camera đang mở -> Đẩy Menu sang trái một khoảng bằng toggleSize
                if (menuRect != null) menuRect.anchoredPosition = new Vector2(-toggleSize, menuRect.anchoredPosition.y);
                StartCoroutine(AutoCaptureRoutine());
            }
            else
            {
                cameraDisplay.gameObject.SetActive(false); 
                // Camera đang tắt -> Hút Menu dính chặt vào mép phải màn hình (X = 0)
                if (menuRect != null) menuRect.anchoredPosition = new Vector2(0, menuRect.anchoredPosition.y);
            }
        }

        isRecording = false;
        if (micButtonBackground != null) micButtonBackground.color = micOffColor;
    }

    void Awake()
    {
        Instance = this;
    }

    void Update()
    {
        if (isRecording && micIcon != null)
        {
            float alpha = 0.6f + Mathf.Sin(Time.time * micBlinkSpeed) * 0.4f;
            micIcon.color = new Color(micIcon.color.r, micIcon.color.g, micIcon.color.b, alpha);
        }
        else if (micIcon != null)
        {
            micIcon.color = new Color(micIcon.color.r, micIcon.color.g, micIcon.color.b, 1f);
        }

        while (mainThreadActions.TryDequeue(out Action action))
        {
            action?.Invoke();
        }
    }

    IEnumerator AutoCaptureRoutine()
    {
        while (true)
        {
            yield return new WaitForSeconds(captureInterval);

            if (webCamTexture != null && webCamTexture.isPlaying)
            {
                CaptureAndSendFace();
            }
        }
    }

    void CaptureAndSendFace()
    {
        if (webCamTexture == null || !webCamTexture.isPlaying) return;

        if (snap == null || snap.width != webCamTexture.width || snap.height != webCamTexture.height)
        {
            snap = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGB24, false);
        }

        snap.SetPixels32(webCamTexture.GetPixels32());
        snap.Apply();

        byte[] imageBytes = snap.EncodeToJPG(40); 
        string imageBase64 = Convert.ToBase64String(imageBytes);

        // --- ĐÃ SỬA: Bắn Base64 thẳng qua file NetworkTest để nó gói JSON và gửi đi ---
        if (NetworkTest.Instance != null)
        {
            NetworkTest.Instance.SendFaceData(imageBase64);
        }
        else
        {
            Debug.LogWarning("Chưa có NetworkTest trong scene!");
        }
    }

    public void AppendToChat(string speaker, string message)
    {
        // Gói lệnh cập nhật UI lại, ném vào Hộp thư
        mainThreadActions.Enqueue(() => 
        {
            if (chatLogText == null) return;
            
            string alignTag = speaker == "User" ? "<align=right>" : "<align=left>";
            
            string colorTag = speaker == "User" ? "<color=#5bc0de>" : "<color=#5cb85c>";
            
            // 3. Ghép tất cả lại (Nhớ có thẻ </align> ở cuối để khóa lề lại nhé)
            chatLogText.text += $"\n{alignTag}{colorTag}{message}</align>\n";

            if (chatScrollRect != null)
            {
                Canvas.ForceUpdateCanvases();
                chatScrollRect.verticalNormalizedPosition = 0f;
            }
        });
    }

    public void ToggleCamera()
    {
        if (webCamTexture == null || cameraDisplay == null) return;

        RectTransform menuRect = actionMenuPanel != null ? actionMenuPanel.GetComponent<RectTransform>() : null;

        if (webCamTexture.isPlaying)
        {
            // TẮT CAM
            webCamTexture.Stop();
            cameraDisplay.gameObject.SetActive(false); 
            // Dời cả Menu Panel và Toggle Button về sát lề phải
            if (menuRect != null) menuRect.anchoredPosition = new Vector2(-10, menuRect.anchoredPosition.y);
            if (toggleMenuButton != null) toggleMenuButton.anchoredPosition = new Vector2(-10, toggleMenuButton.anchoredPosition.y);
        }
        else
        {
            // BẬT CAM
            cameraDisplay.gameObject.SetActive(true); 
            webCamTexture.Play();
            // Đẩy cả Menu Panel và Toggle Button sang trái nhường chỗ cho Cam
            if (menuRect != null) menuRect.anchoredPosition = new Vector2(-toggleSize, menuRect.anchoredPosition.y);
            if (toggleMenuButton != null) toggleMenuButton.anchoredPosition = new Vector2(-toggleSize, toggleMenuButton.anchoredPosition.y);
            
            StopAllCoroutines(); 
            StartCoroutine(AutoCaptureRoutine());
        }
    }

    public void ToggleMicrophone()
    {
        if (currentDeviceMic == null)
        {
            Debug.LogWarning("Không tìm thấy phần cứng Microphone!");
            return;
        }

        if (!isRecording)
        {
            // 1. TRẠNG THÁI: ĐANG TẮT -> CHUYỂN SANG BẬT THU ÂM
            recordingClip = Microphone.Start(currentDeviceMic, false, 300, 16000);
            isRecording = true;

            if (micButtonBackground != null) micButtonBackground.color = micOnColor;
            
            Debug.Log("🔴 Đã BẬT Mic. Đang ghi âm...");
        }
        else
        {
            // 2. TRẠNG THÁI: ĐANG BẬT -> TẮT VÀ GỬI DATA ĐI
            int lastActiveSample = Microphone.GetPosition(currentDeviceMic);
            Microphone.End(currentDeviceMic);
            isRecording = false;

            if (micButtonBackground != null) micButtonBackground.color = micOffColor;
            Debug.Log("⏹ Đã TẮT Mic. Đang xử lý và gửi data...");

            if (lastActiveSample > 0)
            {
                // Đưa qua hàm cắt xén thông minh (Lọc im lặng 2 đầu)
                AudioClip cleanClip = TrimSilence(recordingClip, lastActiveSample);
                Debug.Log($"[Test Trimmer] Độ dài gốc: {recordingClip.length}s | Độ dài sau cắt: {cleanClip.length}s");

                if (cleanClip != null)
                {
                    // Chú ý: Truyền cleanClip.samples vì file đã được cắt gọt vừa vặn
                    byte[] wavBytes = EncodeToWAV(cleanClip, cleanClip.samples);
                    string audioBase64 = Convert.ToBase64String(wavBytes);

                    if (NetworkTest.Instance != null)
                    {
                        NetworkTest.Instance.SendAudioData(audioBase64);
                    }
                    
                }
                else
                {
                    Debug.LogWarning("🔇 File chỉ có tiếng im lặng hoặc quạt xè xè, đã hủy gửi để tiết kiệm API!");
                }
            }
            else
            {
                Debug.LogWarning("Thời gian ghi âm quá ngắn, đã hủy gửi.");
            }
        }
    }

    private AudioClip TrimSilence(AudioClip inputClip, int activeSampleCount, float threshold = 0.02f)
    {
        if (activeSampleCount == 0) return null;

        // Chỉ lấy đúng khoảng thời gian đã thu (bỏ qua đuôi 300s trống)
        float[] samples = new float[activeSampleCount];
        inputClip.GetData(samples, 0); 

        int startSample = 0;
        int endSample = samples.Length - 1;

        // 1. Quét từ đầu tìm điểm có tiếng
        for (int i = 0; i < samples.Length; i++) {
            if (Mathf.Abs(samples[i]) > threshold) {
                startSample = i;
                break;
            }
        }

        // 2. Quét từ cuối ngược lên
        for (int i = samples.Length - 1; i > 0; i--) {
            if (Mathf.Abs(samples[i]) > threshold) {
                endSample = i;
                break;
            }
        }

        // Nếu toàn bộ file là im lặng
        if (startSample >= endSample) return null;

        // 3. Giữ lại 0.2s đệm ở 2 đầu để câu nói tự nhiên, không bị cụt hơi
        int padding = (int)(0.2f * inputClip.frequency * inputClip.channels);
        startSample = Mathf.Max(0, startSample - padding);
        endSample = Mathf.Min(samples.Length - 1, endSample + padding);

        // 4. Tạo file AudioClip mới
        int newSampleCount = endSample - startSample;
        float[] newSamples = new float[newSampleCount];
        Array.Copy(samples, startSample, newSamples, 0, newSampleCount);

        AudioClip trimmedClip = AudioClip.Create("CleanAudio", newSampleCount / inputClip.channels, inputClip.channels, inputClip.frequency, false);
        trimmedClip.SetData(newSamples, 0);

        return trimmedClip;
    }

    private byte[] EncodeToWAV(AudioClip clip, int lastActiveSample)
    {
        using (MemoryStream stream = new MemoryStream())
        {
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                if (lastActiveSample <= 0) lastActiveSample = clip.samples;

                float[] samples = new float[lastActiveSample * clip.channels];
                clip.GetData(samples, 0);

                Int16[] intData = new Int16[samples.Length];
                for (int i = 0; i < samples.Length; i++)
                {
                    intData[i] = (short)(samples[i] * 32767f);
                }

                // Dùng ASCII.GetBytes để khóa chặt chuẩn 1 byte, chống lỗi Header
                writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
                writer.Write(36 + intData.Length * 2);
                writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
                writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
                writer.Write(16);
                writer.Write((short)1);
                writer.Write((short)clip.channels);
                writer.Write(clip.frequency);
                writer.Write(clip.frequency * clip.channels * 2);
                writer.Write((short)(clip.channels * 2));
                writer.Write((short)16);
                writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
                writer.Write(intData.Length * 2);

                byte[] dataBytes = new byte[intData.Length * 2];
                Buffer.BlockCopy(intData, 0, dataBytes, 0, dataBytes.Length);
                writer.Write(dataBytes);
            }
            return stream.ToArray();
        }
    }
}