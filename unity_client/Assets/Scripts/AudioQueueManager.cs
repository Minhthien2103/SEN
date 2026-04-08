using System;
using System.Collections.Generic;
using UnityEngine;

// --- CẤU TRÚC JSON ĐỂ HỨNG TỪ PYTHON ---
[Serializable]
public class EmotionData 
{
    public int emotionId;
    public float emotionIntensity;
}

[Serializable]
public class ServerAudioChunk 
{
    public string event_type;
    public string message_id;
    public int chunk_index;
    public bool is_final;
    public string text_content;
    public string audio_base64;
    public EmotionData emotion_data;
    public List<MouthCue> mouthCues;
}

public class AudioQueueManager: MonoBehaviour
{
    [Header("Thành phần Liên kết")]
    public AudioSource audioSource;
    public Live2DLipSync lipSyncScript;

    // Hàng đợi (Queue) chứa các cục dữ liệu bay về
    private Queue<ServerAudioChunk> chunkQueue = new Queue<ServerAudioChunk>();
    private string currentDeviceMic;
    
    private bool isPlayingChunk = false;

    // Hàm này sau này hệ thống Network (kết nối Python) sẽ gọi khi có data mới
    public void ReceiveChunkFromPython(string jsonString)
    {
        try
        {
            ServerAudioChunk newChunk = JsonUtility.FromJson<ServerAudioChunk>(jsonString);
            chunkQueue.Enqueue(newChunk);
            Debug.Log($"Đã nhận cục Audio thứ: {newChunk.chunk_index}. Đang chờ trong hàng đợi: {chunkQueue.Count}");
        }
        catch (Exception e)
        {
            Debug.LogError("Lỗi giải mã JSON: " + e.Message);
        }
    }

    void Update()
    {
        // Nếu audio đang trống, không có cục nào đang chạy, và hàng đợi có đồ -> Lấy ra phát ngay!
        if (!isPlayingChunk && chunkQueue.Count > 0 && !audioSource.isPlaying)
        {
            PlayNextChunk();
        }
    }

    void PlayNextChunk()
    {
        isPlayingChunk = true;
        ServerAudioChunk currentChunk = chunkQueue.Dequeue();

        if (string.IsNullOrEmpty(currentChunk.audio_base64))
        {
            if (currentChunk.is_final) {
                Debug.Log("🏁 [Trạm Đích] SEN đã nói xong toàn bộ câu!");
            }
            OnChunkFinished(); // Bỏ qua, đọc cục tiếp theo hoặc kết thúc
            return;
        }

        // 1. Dịch chuỗi Base64 thành file âm thanh AudioClip
        AudioClip clip = Base64ToWavAudioClip(currentChunk.audio_base64);

        // --- 🛡️ CẦU CHÌ BẢO VỆ CHỐNG SẬP GAME ---
        if (clip == null)
        {
            Debug.LogWarning($"⚠️ Cục audio số {currentChunk.chunk_index} bị lỗi định dạng, bỏ qua!");
            OnChunkFinished(); // Cho phép hệ thống bỏ qua và đọc cục tiếp theo
            return;
        }
        // ----------------------------------------

        // 2. Nạp dữ liệu cảm xúc và khẩu hình cho Live2D
        lipSyncScript.PrepareNewChunk(currentChunk.mouthCues, currentChunk.emotion_data.emotionId, currentChunk.emotion_data.emotionIntensity);

        // 3. Phát âm thanh
        audioSource.clip = clip;
        audioSource.Play();

        // 4. Hẹn giờ khi nào đọc xong cục này thì cho phép đọc cục tiếp theo
        Invoke("OnChunkFinished", clip.length + 0.1f); 
    }

    void OnChunkFinished()
    {
        isPlayingChunk = false;
        // Nếu hàng đợi đã hết và đây là cục cuối cùng (is_final = true), ta có thể kích hoạt trạng thái Idle/Listening cho AI.
    }

    // --- CÔNG CỤ DỊCH BASE64 SANG AUDIO (Giả định Python gửi chuẩn WAV 16-bit PCM) ---
    private AudioClip Base64ToWavAudioClip(string base64String)
    {
        byte[] wavBytes = Convert.FromBase64String(base64String);

        // Quét tìm vị trí bắt đầu của chunk "data" thật sự trong file WAV
        int dataIndex = -1;
        for (int i = 12; i < wavBytes.Length - 4; i++)
        {
            if (wavBytes[i] == 'd' && wavBytes[i + 1] == 'a' && wavBytes[i + 2] == 't' && wavBytes[i + 3] == 'a')
            {
                dataIndex = i;
                break;
            }
        }

        if (dataIndex == -1) 
        {
            Debug.LogError("Không tìm thấy vùng chứa âm thanh trong file WAV!");
            return null;
        }

        // Đọc thông số chuẩn xác bất chấp độ dài Header
        int channels = BitConverter.ToInt16(wavBytes, 22);
        int sampleRate = BitConverter.ToInt32(wavBytes, 24);
        int dataSize = BitConverter.ToInt32(wavBytes, dataIndex + 4);
        int audioStartIndex = dataIndex + 8;

        // An toàn chống tràn bộ nhớ (Overflow)
        int sampleCount = dataSize / 2;
        float[] audioData = new float[sampleCount];
        
        for (int i = 0; i < sampleCount; i++)
        {
            // Kiểm tra ranh giới mảng để tránh lỗi văng game
            if (audioStartIndex + i * 2 + 1 < wavBytes.Length)
            {
                audioData[i] = BitConverter.ToInt16(wavBytes, audioStartIndex + i * 2) / 32768f;
            }
        }

        AudioClip clip = AudioClip.Create("StreamingChunk", sampleCount, channels, sampleRate, false);
        clip.SetData(audioData, 0);
        return clip;
    }

    void Start()
    {
        // 1. Kiểm tra xem máy có cắm mic không
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("❌ CHẾT DỞ: Unity không tìm thấy cái Microphone nào trên máy!");
            return;
        }

        // 2. In ra toàn bộ danh sách Mic mà Unity đọc được
        Debug.Log("🎤 TỔNG HỢP DANH SÁCH MICROPHONE ĐANG KẾT NỐI:");
        for (int i = 0; i < Microphone.devices.Length; i++)
        {
            Debug.Log($"[{i}] {Microphone.devices[i]}");
        }

        // 3. Hiển thị cái Mic mà code đang chọn để thu âm
        // Giả sử biến lưu tên mic của sếp tên là currentDeviceMic
        currentDeviceMic = Microphone.devices[0]; 
        Debug.Log($"👉 [QUAN TRỌNG] Unity đang chọn dùng Mic: {currentDeviceMic}");
    }
}