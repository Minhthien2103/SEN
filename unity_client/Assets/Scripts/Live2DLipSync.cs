using System.Collections.Generic;
using UnityEngine;
using Live2D.Cubism.Core;

[System.Serializable]
public class MouthCue
{
    public float start;
    public float end;
    public string value; 
}

public enum AIState { Idle, Listening, Thinking, Speaking }

public class Live2DLipSync : MonoBehaviour
{
    [Header("Thành phần hệ thống")]
    public AudioSource audioSource;

    private List<MouthCue> currentCues = new List<MouthCue>();
    private CubismModel model;

    private CubismParameter breathParam;
    private CubismParameter mouthOpenParam, mouthFormParam;
    private CubismParameter eyeLOpen, eyeROpen, eyeLSmile, eyeRSmile;
    private CubismParameter browLY, browRY, browLAngle, browRAngle, browLForm, browRForm;
    private CubismParameter cheekParam, headAngleX, headAngleY, headAngleZ;
    
    // Normalization constant
    private const float HEAD_ANGLE_NORMALIZE = 30f;

    private float targetMouthOpen = 0f, targetMouthForm = 0f;
    private float targetEyeLOpen = 1f, targetEyeROpen = 1f, targetEyeSmile = 0f;
    private float targetBrowLY = 0f, targetBrowRY = 0f, targetBrowLAngle = 0f, targetBrowRAngle = 0f, targetBrowLForm = 0f, targetBrowRForm = 0f;
    private float targetCheek = 0f, targetHeadX = 0f, targetHeadY = 0f, targetHeadZ = 0f;

    private float curMouthOpen = 0f, curMouthForm = 0f;
    private float curEyeLOpen = 1f, curEyeROpen = 1f, curEyeSmile = 0f;
    private float curBrowLY = 0f, curBrowRY = 0f, curBrowLAngle = 0f, curBrowRAngle = 0f, curBrowLForm = 0f, curBrowRForm = 0f;
    private float curCheek = 0f, curHeadX = 0f, curHeadY = 0f, curHeadZ = 0f;

    private float baseEmotionMouthForm = 0f;
    private float baseEmotionMouthOpen = 0f; 

    [Header("Cài đặt Tốc độ")]
    public float emotionTransitionSpeed = 5f;
    public float mouthOpenSpeed = 12f;  // Mở từ từ, mềm mại
    public float mouthCloseSpeed = 30f; // Khép lại mạnh (bập môi)
    public float mouthFormSpeed = 15f;  // Form chuyển nhanh hơn (cho viseme chuyển tiếp) 

    [Header("Trạng thái AI")]
    public AIState currentState = AIState.Idle;
    [Range(0, 2)]
    public int currentPoseVariation = 0;
    private float stateTimer = 0f;
    private float poseChangeTimer = 0f; 
    
    [Range(0, 6)]
    public int currentEmotionId = 6; 
    private float currentEmotionIntensity = 1.0f; // Mặc định 100%

    private float blinkTimer = 3f;
    private float blinkMultiplier = 1f;
    private float[] audioSamples = new float[256];
    private bool suppressBlink = false; // Tắt blink khi Surprise/Fear

    private float breathCycle = 0f;
    private float currentVolumeRMS = 0f;
    
    [Header("Tùy chọn")]
    public bool breathingEnabled = true;  // Toggle breathing effect
    public float speechBobbingIntensity = 6f;  // Adjust how much speech affects head movement

    void Start()
    {
        model = this.GetComponent<CubismModel>();
        if (model != null)
        {
            mouthOpenParam = model.Parameters.FindById("ParamMouthOpenY");
            mouthFormParam = model.Parameters.FindById("ParamMouthForm");
            eyeLOpen = model.Parameters.FindById("ParamEyeLOpen");
            eyeROpen = model.Parameters.FindById("ParamEyeROpen");
            eyeLSmile = model.Parameters.FindById("ParamEyeLSmile");
            eyeRSmile = model.Parameters.FindById("ParamEyeRSmile");
            browLY = model.Parameters.FindById("ParamBrowLY");
            browRY = model.Parameters.FindById("ParamBrowRY");
            browLAngle = model.Parameters.FindById("ParamBrowLAngle");
            browRAngle = model.Parameters.FindById("ParamBrowRAngle");
            browLForm = model.Parameters.FindById("ParamBrowLForm");
            browRForm = model.Parameters.FindById("ParamBrowRForm");
            cheekParam = model.Parameters.FindById("ParamCheek");

            headAngleX = model.Parameters.FindById("ParamAngleX"); // Lắc đầu trái phải
            headAngleY = model.Parameters.FindById("ParamAngleY"); 
            headAngleZ = model.Parameters.FindById("ParamAngleZ"); 
            
            breathParam = model.Parameters.FindById("ParamBreath"); // Hơi thở
        }

        if (audioSource != null)
        {
            audioSource.Play();
        }
    }

    public void PrepareNewChunk(List<MouthCue> newCues, int emotionId, float emotionIntensity)
    {
        currentCues = newCues ?? new List<MouthCue>();
        currentEmotionId = emotionId;
        currentEmotionIntensity = Mathf.Clamp01(emotionIntensity);        
    }

    public void SetAIState(AIState newState)
    {
        if (currentState == newState)
        {
            return;
        }
        
        currentState = newState;
        stateTimer = 0f;
        poseChangeTimer = 0f;
        
        // Random ngay một tư thế mới khi vừa vào trạng thái
        currentPoseVariation = Random.Range(0, 3); // [0, 1, 2]
    }

    void Update()
    {
        // --- 1. HỆ THỐNG HƠI THỞ (BREATHING VITALITY) ---
        if (breathingEnabled)
        {
            breathCycle += Time.deltaTime * 2.5f; 
        } 

        currentVolumeRMS = 0f;
        if (audioSource.isPlaying)
        {
            audioSource.GetOutputData(audioSamples, 0);
            float sum = 0f;
            foreach (float sample in audioSamples) sum += sample * sample;
            float rms = Mathf.Sqrt(sum / 256f);
            currentVolumeRMS = Mathf.Clamp(rms * 10.0f, 0f, 1f);
        }

        // Cập nhật State Machine
        ApplyEmotionShape(currentEmotionId, currentEmotionIntensity);
        
        ProcessAIStates();

        // --- 2. VI CHUYỂN ĐỘNG (MICRO-MOVEMENTS) ---
        // Dùng Perlin noise để tạo dao động lắc lư cổ cực nhỏ
        float noiseX = (Mathf.PerlinNoise(Time.time * 0.3f, 0f) - 0.5f) * 0.1f;  // Giới hạn ±0.05f
        float noiseZ = (Mathf.PerlinNoise(0f, Time.time * 0.3f) - 0.5f) * 0.08f;  // Giới hạn ±0.04f
        
        // Thêm noise vào emotion base (không tích lũy)
        targetHeadX = noiseX;
        targetHeadZ += noiseZ;

        // --- 3. AUTO-BLINK CẢI TIẾN ---
        blinkTimer -= Time.deltaTime;
        if (blinkTimer <= 0f)
        {
            blinkMultiplier = 0f; 
            // VTuber pro chớp mắt thường có 15% cơ hội nháy đúp (Double blink)
            if (Random.value > 0.85f) {
                blinkTimer = 0.2f;  // Nháy lần 2 sau 0.2 giây
            } else {
                blinkTimer = Random.Range(3.0f, 6.0f);
            } 
        }
        blinkMultiplier = Mathf.Lerp(blinkMultiplier, 1f, Time.deltaTime * 30f);

        if (audioSource.isPlaying)
        {
            targetHeadY = Mathf.Clamp(targetHeadY + (currentVolumeRMS * speechBobbingIntensity / HEAD_ANGLE_NORMALIZE), -1f, 1f);
        }

        // --- 5. NỘI SUY MƯỢT CÁC KHỚP ---
        curEyeLOpen = Mathf.Lerp(curEyeLOpen, targetEyeLOpen, Time.deltaTime * emotionTransitionSpeed);
        curEyeROpen = Mathf.Lerp(curEyeROpen, targetEyeROpen, Time.deltaTime * emotionTransitionSpeed);
        curEyeSmile = Mathf.Lerp(curEyeSmile, targetEyeSmile, Time.deltaTime * emotionTransitionSpeed);
        
        curBrowLY = Mathf.Lerp(curBrowLY, targetBrowLY, Time.deltaTime * emotionTransitionSpeed);
        curBrowRY = Mathf.Lerp(curBrowRY, targetBrowRY, Time.deltaTime * emotionTransitionSpeed);
        curBrowLAngle = Mathf.Lerp(curBrowLAngle, targetBrowLAngle, Time.deltaTime * emotionTransitionSpeed);
        curBrowRAngle = Mathf.Lerp(curBrowRAngle, targetBrowRAngle, Time.deltaTime * emotionTransitionSpeed);
        curBrowLForm = Mathf.Lerp(curBrowLForm, targetBrowLForm, Time.deltaTime * emotionTransitionSpeed);
        curBrowRForm = Mathf.Lerp(curBrowRForm, targetBrowRForm, Time.deltaTime * emotionTransitionSpeed);
        curCheek = Mathf.Lerp(curCheek, targetCheek, Time.deltaTime * emotionTransitionSpeed);
        
        curHeadX = Mathf.Lerp(curHeadX, targetHeadX, Time.deltaTime * emotionTransitionSpeed * 0.4f);
        curHeadY = Mathf.Lerp(curHeadY, targetHeadY, Time.deltaTime * emotionTransitionSpeed * 0.5f); 
        curHeadZ = Mathf.Lerp(curHeadZ, targetHeadZ, Time.deltaTime * emotionTransitionSpeed * 0.5f);
        
        // Clamp final head values to valid range
        curHeadX = Mathf.Clamp(curHeadX, -1f, 1f);
        curHeadY = Mathf.Clamp(curHeadY, -1f, 1f);
        curHeadZ = Mathf.Clamp(curHeadZ, -1f, 1f);

        // --- 6. XỬ LÝ KHẨU HÌNH (LIP-SYNC TỐI THƯỢNG) ---
        ProcessLipSync();
    }

    void ProcessAIStates()
    {
        // Nếu đang nói (Audio Source is playing), tự động chuyển sang Speaking và thoát hàm
        if (audioSource.isPlaying) 
        {
            currentState = AIState.Speaking;
            return; 
        }

        stateTimer += Time.deltaTime;
        poseChangeTimer += Time.deltaTime;

        // Cứ mỗi 3 đến 5 giây, AI sẽ tự động đổi tư thế một lần cho đỡ cứng người
        if (poseChangeTimer > Random.Range(3f, 5f) && currentState != AIState.Idle)
        {
            currentPoseVariation = Random.Range(0, 3); // Bốc ngẫu nhiên kiểu 0, 1 hoặc 2
            poseChangeTimer = 0f; // Reset đồng hồ đổi tư thế
        }

        switch (currentState)
        {
            case AIState.Listening:
                ApplyListeningVariations(currentPoseVariation);
                break;

            case AIState.Thinking:
                ApplyThinkingVariations(currentPoseVariation);
                break;

            case AIState.Idle:
                // Idle thì để Perlin Noise tự xử lý lắc lư, mặt về Normal
                targetHeadX = targetHeadY = targetHeadZ = 0f;
                targetBrowLY = targetBrowRY = 0f;
                targetMouthForm = 0f;
                break;
                
            case AIState.Speaking:
                // Do ApplyEmotionShape lo
                break;
        }
    }

    void ApplyListeningVariations(int variation)
    {
        float blend = 0.8f; 

        targetEyeLOpen = Mathf.Lerp(targetEyeLOpen, 1.15f, blend);
        targetEyeROpen = Mathf.Lerp(targetEyeROpen, 1.15f, blend);

        // TUYỆT KỸ: Gật đầu cực sâu (Cúi gập 100% biên độ)
        float nodWave = 0f;
        float nodCycle = stateTimer % 4f; // Chu kỳ 4 giây
        if (nodCycle > 2.5f) 
        {
            // Lấy 1.5 giây cuối của chu kỳ để gật
            float timeInNod = nodCycle - 2.5f; 
            
            // Mathf.Abs giúp sóng Sin luôn dương -> nhân với số âm sẽ luôn gập cổ XUỐNG
            // Nâng biên độ lên tận -30f để xuyên thủng lớp "giảm xóc" của hàm Lerp
            nodWave = Mathf.Abs(Mathf.Sin(timeInNod * Mathf.PI * 1.5f)) * -30f / HEAD_ANGLE_NORMALIZE;
        }

        switch (variation)
        {
            case 0: 
                targetHeadZ = Mathf.Lerp(targetHeadZ, 18f / HEAD_ANGLE_NORMALIZE, blend); 
                targetHeadX = Mathf.Lerp(targetHeadX, -8f / HEAD_ANGLE_NORMALIZE, blend);
                targetBrowLY = Mathf.Lerp(targetBrowLY, 0.4f, blend); 
                targetBrowRY = Mathf.Lerp(targetBrowRY, 0.6f, blend); 
                targetMouthForm = Mathf.Lerp(targetMouthForm, 0.3f, blend); 
                break;
                
            case 1: 
                targetHeadY = Mathf.Lerp(targetHeadY, -8f / HEAD_ANGLE_NORMALIZE, blend);
                targetHeadZ = Mathf.Lerp(targetHeadZ, -2f / HEAD_ANGLE_NORMALIZE, blend);
                targetHeadX = Mathf.Lerp(targetHeadX, 10f / HEAD_ANGLE_NORMALIZE, blend);
                targetEyeSmile = Mathf.Lerp(targetEyeSmile, 0.6f, blend); 
                targetMouthForm = Mathf.Lerp(targetMouthForm, 0.5f, blend); 
                break;

            case 2: 
                targetHeadY = Mathf.Lerp(targetHeadY, 5f / HEAD_ANGLE_NORMALIZE, blend); 
                targetHeadZ = Mathf.Lerp(targetHeadZ, 5f / HEAD_ANGLE_NORMALIZE, blend);
                targetBrowLY = Mathf.Lerp(targetBrowLY, -0.3f, blend); 
                targetBrowRY = Mathf.Lerp(targetBrowRY, -0.3f, blend); 
                targetMouthForm = Mathf.Lerp(targetMouthForm, 0.0f, blend);
                break;
        }
        
        // Trộn nhịp gật gù siêu mạnh này vào trục Y
        targetHeadY += nodWave; 
    }

    void ApplyThinkingVariations(int variation)
    {
        // Khi suy nghĩ, tư thế phải lấn át hoàn toàn (90%) để thấy rõ sự ngắt quãng giao tiếp
        float blend = 0.9f; 
        
        // Mắt hơi nheo lại, tròng mắt đờ ra
        targetEyeLOpen = Mathf.Lerp(targetEyeLOpen, 0.85f, blend);
        targetEyeROpen = Mathf.Lerp(targetEyeROpen, 0.85f, blend);

        switch (variation)
        {
            case 0: // Kiểu "Loading": Ngước mạnh lên trời, liếc sang góc, chu môi
                targetHeadY = Mathf.Lerp(targetHeadY, 20f / HEAD_ANGLE_NORMALIZE, blend); // Ngước rất cao
                targetHeadX = Mathf.Lerp(targetHeadX, 20f / HEAD_ANGLE_NORMALIZE, blend); // Ngoảnh sang trái
                targetHeadZ = Mathf.Lerp(targetHeadZ, 5f / HEAD_ANGLE_NORMALIZE, blend);
                targetBrowLY = Mathf.Lerp(targetBrowLY, 0.5f, blend);
                targetBrowRY = Mathf.Lerp(targetBrowRY, 0.8f, blend); // Rướn 1 bên mày
                targetMouthForm = Mathf.Lerp(targetMouthForm, -0.6f, blend); // Mím/chu môi
                break;

            case 1: // Kiểu "Bí ý tưởng": Cúi gằm mặt xuống phải, mày nhíu chặt căng thẳng
                targetHeadY = Mathf.Lerp(targetHeadY, -18f / HEAD_ANGLE_NORMALIZE, blend);
                targetHeadX = Mathf.Lerp(targetHeadX, -15f / HEAD_ANGLE_NORMALIZE, blend);
                targetHeadZ = Mathf.Lerp(targetHeadZ, -10f / HEAD_ANGLE_NORMALIZE, blend);
                targetBrowLY = Mathf.Lerp(targetBrowLY, -0.8f, blend); // Mày nhíu gắt
                targetBrowRY = Mathf.Lerp(targetBrowRY, -0.8f, blend);
                targetBrowLAngle = Mathf.Lerp(targetBrowLAngle, -0.5f, blend);
                targetBrowRAngle = Mathf.Lerp(targetBrowRAngle, -0.5f, blend);
                targetMouthForm = Mathf.Lerp(targetMouthForm, -0.8f, blend); // Cắn môi
                break;

            case 2: // Kiểu "À há": Đầu nghiêng sâu, mắt mở to hơn xíu chớp liên tục
                targetHeadZ = Mathf.Lerp(targetHeadZ, -22f / HEAD_ANGLE_NORMALIZE, blend); // Nghiêng tít sang bên
                targetHeadY = Mathf.Lerp(targetHeadY, 10f / HEAD_ANGLE_NORMALIZE, blend);
                targetHeadX = Mathf.Lerp(targetHeadX, -5f / HEAD_ANGLE_NORMALIZE, blend);
                targetEyeLOpen = Mathf.Lerp(targetEyeLOpen, 1.2f, blend); 
                targetEyeROpen = Mathf.Lerp(targetEyeROpen, 1.2f, blend);
                targetBrowLY = Mathf.Lerp(targetBrowLY, 0.7f, blend);
                targetBrowRY = Mathf.Lerp(targetBrowRY, 0.7f, blend);
                targetMouthForm = Mathf.Lerp(targetMouthForm, 0.3f, blend); // Hé miệng nhẹ
                break;
        }
    }

    void ProcessLipSync()
    {
        targetMouthOpen = baseEmotionMouthOpen; 
        targetMouthForm = baseEmotionMouthForm;
        
        // SỬA Ở ĐÂY: Chỉ đóng miệng khi Audio thực sự KHÔNG phát
        if (!audioSource.isPlaying) 
        {
            curMouthOpen = Mathf.Lerp(curMouthOpen, targetMouthOpen, Time.deltaTime * mouthCloseSpeed);
            curMouthForm = Mathf.Lerp(curMouthForm, targetMouthForm, Time.deltaTime * mouthFormSpeed);
            return;
        }

        float volumeMultiplier = 0.4f + (currentVolumeRMS * 0.8f); 
        float rms = currentVolumeRMS / 10.0f; 
        float time = audioSource.time;

        // PHƯƠNG ÁN DỰ PHÒNG: Nếu có Audio mà chưa có JSON, nhép miệng bằng Âm lượng!
        if (currentCues == null || currentCues.Count == 0)
        {
            // Nhân RMS để miệng đập nhịp nhàng theo tiếng nói
            targetMouthOpen = currentVolumeRMS * 1.5f; 
            targetMouthForm = baseEmotionMouthForm;
        }
        else
        {
            foreach (var cue in currentCues)
            {
                if (time >= cue.start && time <= cue.end)
                {
                    switch (cue.value)
                    {
                        case "A": targetMouthOpen = 0.0f;  break; 
                        case "B": targetMouthOpen = 0.15f;  break; 
                        case "C": targetMouthOpen = 0.35f; targetMouthForm = 1.0f;  break; 
                        case "D": targetMouthOpen = 1.0f;  targetMouthForm = 0.2f;  break; 
                        case "E": targetMouthOpen = 0.5f;  targetMouthForm = -1.0f; break; 
                        case "F": targetMouthOpen = 0.2f;  targetMouthForm = -1.0f; break; 
                        case "G": targetMouthOpen = 0.2f;  targetMouthForm = 0.4f;  break; 
                        case "H": targetMouthOpen = 0.3f;  targetMouthForm = 0.1f;  break; 
                        case "X": targetMouthOpen = 0.0f;  break; 
                    }

                    targetMouthOpen *= volumeMultiplier;

                    float formVariation = (rms * 0.08f) * Mathf.Sign(baseEmotionMouthForm);
                    targetMouthForm = Mathf.Clamp(targetMouthForm + formVariation, -1f, 1f);
                    break;
                }
            }
        }
        
        float currentDynamicSpeed = (targetMouthOpen < curMouthOpen) ? mouthCloseSpeed : mouthOpenSpeed;
        curMouthOpen = Mathf.Lerp(curMouthOpen, targetMouthOpen, Time.deltaTime * currentDynamicSpeed);
        curMouthForm = Mathf.Lerp(curMouthForm, targetMouthForm, Time.deltaTime * mouthFormSpeed);
    }

    void LateUpdate()
    {   
        if (mouthOpenParam != null) mouthOpenParam.Value = curMouthOpen;
        if (mouthFormParam != null) mouthFormParam.Value = curMouthForm;

        // Áp dụng nháy mắt, nếu đang ngạc nhiên hoặc hoảng sợ thì mắt căng ra khó nháy hơn
        float finalBlink = blinkMultiplier;
        if (suppressBlink) finalBlink = Mathf.Max(0.5f, blinkMultiplier);

        if (eyeLOpen != null) eyeLOpen.Value = curEyeLOpen * finalBlink;
        if (eyeROpen != null) eyeROpen.Value = curEyeROpen * finalBlink;
        if (eyeLSmile != null) eyeLSmile.Value = curEyeSmile;
        if (eyeRSmile != null) eyeRSmile.Value = curEyeSmile;
        if (cheekParam != null) cheekParam.Value = curCheek;

        if (browLY != null) browLY.Value = curBrowLY;
        if (browRY != null) browRY.Value = curBrowRY;
        if (browLAngle != null) browLAngle.Value = curBrowLAngle;
        if (browRAngle != null) browRAngle.Value = curBrowRAngle;
        if (browLForm != null) browLForm.Value = curBrowLForm;
        if (browRForm != null) browRForm.Value = curBrowRForm;

        if (headAngleX != null) headAngleX.Value = curHeadX;
        if (headAngleY != null) headAngleY.Value = curHeadY;
        if (headAngleZ != null) headAngleZ.Value = curHeadZ;

        // Truyền nhịp thở vào model (Biến thiên từ 0 -> 1)
        if (breathParam != null) breathParam.Value = (Mathf.Sin(breathCycle) + 1f) * 0.5f;
    }

    void ApplyEmotionShape(int emotion, float intensity)
    {
        targetEyeLOpen = targetEyeROpen = 1.0f; 
        targetEyeSmile = 0f;
        targetBrowLY = targetBrowRY = 0f;
        targetBrowLAngle = targetBrowRAngle = 0f;
        targetBrowLForm = targetBrowRForm = 0f;
        targetCheek = 0f;
        targetHeadX = 0f; targetHeadY = 0f; targetHeadZ = 0f;
        baseEmotionMouthForm = 0f;
        baseEmotionMouthOpen = 0f;
        suppressBlink = false; // Mặc định cho phép blink 

        switch (emotion)
        {
            case 0: // Enjoyment (Cười hạnh phúc - Crow's feet + cheek lift + warm tilt)
                targetEyeLOpen = targetEyeROpen = 1.0f - (0.15f * intensity);  // Nhắm tít lại hình vầng trăng (genuine smile)
                targetEyeSmile = 1.0f * intensity;  // Crow's feet rõ rệt
                targetBrowLY = targetBrowRY = 0.4f * intensity;  // Chân mày nhẹ nhàng nâng lên
                targetBrowLAngle = targetBrowRAngle = 0.3f * intensity;  // Không quá gắt
                targetBrowLForm = targetBrowRForm = 0.8f * intensity;  // Mềm mại
                
                targetCheek = 1.0f * intensity; 
                targetHeadZ = (10f * intensity) / HEAD_ANGLE_NORMALIZE; 
                targetHeadY = (3f * intensity) / HEAD_ANGLE_NORMALIZE;  // Nghiêng + hơi ngửa
                baseEmotionMouthForm = 0.9f * intensity;  // Cười tươi rói
                break;
                
            case 1: // Sadness (Buồn bã - Đầu mày dướn lên, đuôi cụp xuống, miệng buồn)
                targetEyeLOpen = targetEyeROpen = 1.0f - (0.3f * intensity);  // Mắt rũ, không mở
                targetBrowLY = 0.4f * intensity;   // Chân mày trong (inner) nâng lên
                targetBrowRY = -0.2f * intensity;  // Chân mày ngoài (outer) hạ xuống
                targetBrowLAngle = -0.6f * intensity;
                targetBrowRAngle = -0.8f * intensity;  // Góc sad (V ngược)
                targetBrowLForm = targetBrowRForm = -1.0f * intensity;

                targetCheek = 0.0f * intensity;  // Trắng nhợt, không máu
                targetHeadY = (-18f * intensity) / HEAD_ANGLE_NORMALIZE;  // Cúi sâu hơn
                targetHeadZ = (3f * intensity) / HEAD_ANGLE_NORMALIZE;
                baseEmotionMouthForm = -0.7f * intensity;  // Không chu
                break;

            case 2: // Fear (Sợ hãi - Mắt mở tròn, miệng căng, nhét lưỡi)
                targetEyeLOpen = targetEyeROpen = 1.0f + (0.4f * intensity);  // Mắt mở nhưng tự nhiên hơn
                targetBrowLY = 0.5f * intensity; 
                targetBrowRY = 0.5f * intensity;  // Cả hai nâng lên (tense)
                targetBrowLAngle = -0.4f * intensity; 
                targetBrowRAngle = -0.6f * intensity;  // Hơi nhíu
                targetBrowLForm = targetBrowRForm = -0.5f * intensity;
                
                targetHeadX = (10f * intensity) / HEAD_ANGLE_NORMALIZE;  // Lắc trái phải (normalized)
                targetHeadY = (8f * intensity) / HEAD_ANGLE_NORMALIZE; 
                targetHeadZ = (-8f * intensity) / HEAD_ANGLE_NORMALIZE;  // Lùi đầu + lắc
                baseEmotionMouthForm = -0.6f * intensity;  // Căng lại (như "aahhh")
                suppressBlink = true;  // Không chớp mắt khi sợ hãi
                break;

            case 3: // Anger (Tức giận - Mắt sắc, chân mày chữ V, miệng nhếch)
                targetEyeLOpen = targetEyeROpen = 1.0f - (0.25f * intensity);  // Mắt rất hẹp, sắc bén
                targetBrowLY = targetBrowRY = -0.85f * intensity;  // Chân mày quặp xuống cực mạnh
                targetBrowLAngle = targetBrowRAngle = -1.0f * intensity;  // Chữ V gắt
                targetBrowLForm = targetBrowRForm = -1.0f * intensity;
                
                targetCheek = 0.6f * intensity;  // Đỏ mặt (máu lên)
                targetHeadY = (-8f * intensity) / HEAD_ANGLE_NORMALIZE;  // Chúi đầu tới
                targetHeadZ = (2f * intensity) / HEAD_ANGLE_NORMALIZE;  // Mó nhẹ lên
                baseEmotionMouthForm = -0.8f * intensity;  // Nhếch mép (snarl)
                break;

            case 4: // Disgust
                targetEyeLOpen = 1.0f - (0.4f * intensity); 
                targetEyeROpen = 1.0f - (0.1f * intensity); 
                targetBrowLY = -0.7f * intensity; targetBrowRY = 0.4f * intensity; 
                targetBrowLAngle = -0.8f * intensity; targetBrowRAngle = 0.6f * intensity; 
                targetBrowLForm = -0.8f * intensity; targetBrowRForm = -0.2f * intensity;

                targetHeadX = (-15f * intensity) / HEAD_ANGLE_NORMALIZE;  // Normalized to -0.5f
                targetHeadY = (2f * intensity) / HEAD_ANGLE_NORMALIZE; 
                targetHeadZ = (-10f * intensity) / HEAD_ANGLE_NORMALIZE; 
                baseEmotionMouthForm = -0.5f * intensity;
                break;

            case 5: // Surprise (Ngạc nhiên - Mắt mở hết cỡ, rươi lên, rớt hàm)
                targetEyeLOpen = targetEyeROpen = 1.0f + (0.4f * intensity); // Mắt mở to, tự nhiên
                targetBrowLY = targetBrowRY = 1.0f * intensity; // RƠI LÔNG MÀY LÊN CỰC CAO
                targetBrowLAngle = targetBrowRAngle = 0.2f * intensity; // Hơi vổ (surprise angle)
                targetBrowLForm = targetBrowRForm = 0.8f * intensity;
                
                targetHeadY = (12f * intensity) / HEAD_ANGLE_NORMALIZE; // Ngửa hẳn mặt lên
                baseEmotionMouthOpen = 0.25f * intensity; // RỚT HÀM ro rõ
                baseEmotionMouthForm = 0.0f * intensity; // Tròn (O shape)
                suppressBlink = true; // Không chớp mắt khi ngạc nhiên
                break;

            case 6: // Neutral
            default:
                targetHeadZ = (1f * intensity) / HEAD_ANGLE_NORMALIZE; 
                break;
        }
    }
}