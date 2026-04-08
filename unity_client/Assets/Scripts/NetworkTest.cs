using System;
using System.Collections.Generic; 
using UnityEngine;
using SocketIOClient; 

public class NetworkTest : MonoBehaviour
{
    // Tạo Singleton để AppInputManager dễ dàng gọi ké
    public static NetworkTest Instance;

    [Header("Cấu hình Mạng")]
    [Tooltip("Dán link Ngrok HTTPS vào đây, KHÔNG có dấu gạch chéo / ở cuối")]
    public string serverURL = "https://brushable-lamont-palaeontographic.ngrok-free.dev"; 

    public AudioQueueManager audioQueue;

    private SocketIOUnity client;

    void Awake()
    {
        Instance = this;
    }

    void Start()
    {
        // Truyền thẳng đường link Ngrok vào Uri
        var uri = new Uri(serverURL);
        client = new SocketIOUnity(uri, new SocketIOOptions
        {
            Transport = SocketIOClient.Transport.TransportProtocol.WebSocket
        });

        client.OnConnected += (sender, e) => Debug.Log("✅ [Network] ĐÃ KẾT NỐI TỚI PYTHON (QUA NGROK)!");

        // 1. Hứng STT (Giọng nói của sếp)
        client.On("server_user_text", response =>
        {
            string rawJson = response.ToString().Trim();
            if (rawJson.StartsWith("[")) rawJson = rawJson.Substring(1, rawJson.Length - 2);
            
            var data = JsonUtility.FromJson<ChatPayload>(rawJson); 
            
            // GỌI CHỦ NHÀ: Báo là User vừa nói
            if (AppInputManager.Instance != null && !string.IsNullOrEmpty(data.text))
            {
                AppInputManager.Instance.AppendToChat("User", data.text);
            }
        });

        // 2. Hứng câu trả lời của SEN
        client.On("server_text_reply", response =>
        {
            string rawJson = response.ToString().Trim();
            if (rawJson.StartsWith("[")) rawJson = rawJson.Substring(1, rawJson.Length - 2);
            
            var data = JsonUtility.FromJson<ChatPayload>(rawJson); 
            
            // GỌI CHỦ NHÀ: Báo là SEN vừa nói
            if (AppInputManager.Instance != null && !string.IsNullOrEmpty(data.message))
            {
                AppInputManager.Instance.AppendToChat("SEN", data.message);
            }
        });
        
        // 3. Hứng Audio Chunk (Giữ nguyên)
        client.On("server_audio_chunk", response =>
        {
            string rawJson = response.ToString().Trim();
            if (rawJson.StartsWith("[")) rawJson = rawJson.Substring(1, rawJson.Length - 2);

            Debug.Log("📥 [Network] ĐÃ NHẬN JSON AUDIO TỪ PYTHON!");

            if (audioQueue != null)
            {
                audioQueue.ReceiveChunkFromPython(rawJson);
            }
            else
            {
                Debug.LogError("Chưa kéo file AudioQueueManager vào NetworkTest!");
            }
        });

        client.Connect();
    }

    public void SendFaceData(string base64Image)
    {
        if (client != null && client.Connected)
        {
            var jsonData = new Dictionary<string, object>
            {
                { "user_id", "uid_12345" },
                { "session_id", "sess_001" },
                { "image_base64", base64Image } 
            };
            client.Emit("client_face_input", jsonData);
            Debug.Log("📤 [Network] Đã bắn Gói tin Ảnh (client_face_input) đi!");
        }
    }

    public void SendAudioData(string base64Audio)
    {
        if (client != null && client.Connected)
        {
            var jsonData = new Dictionary<string, object>
            {
                { "user_id", "uid_12345" },
                { "session_id", "sess_001" },
                { "audio_base64", base64Audio },
                { "is_end_of_speech", true }
            };
            client.Emit("client_audio_input", jsonData);
            Debug.Log("📤 [Network] Đã bắn Gói tin Âm thanh (client_audio_input) đi!");
        }
    }

    void OnDestroy()
    {
        if (client != null)
        {
            client.Disconnect();
            client.Dispose();
        }
    }
}


[System.Serializable]
public class ChatPayload
{
    public string text;    // (STT)
    public string message; // (SEN trả lời)
}