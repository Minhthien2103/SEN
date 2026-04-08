using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class WindowController : MonoBehaviour
{
    // --- KHAI BÁO THƯ VIỆN CỦA WINDOWS ĐỂ LÀM NÚT MINIMIZE ---
    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hwnd, int nCmdShow);
    [DllImport("user32.dll")]
    private static extern IntPtr GetActiveWindow();

    // 1. NÚT THU NHỎ (-)
    public void MinimizeWindow()
    {
        // Gọi thẳng lệnh thu nhỏ của hệ điều hành Windows
        ShowWindow(GetActiveWindow(), 2); 
    }

    // 2. NÚT PHÓNG TO / THU HẸP (Square)
    public void ToggleMaximize()
    {
        Screen.fullScreen = !Screen.fullScreen;
    }

    // 3. NÚT TẮT APP (X)
    public void CloseApp()
    {
        Debug.Log("Đang tắt App...");
        Application.Quit(); 
        // Lưu ý: Lệnh Quit() chỉ có tác dụng khi đã Build ra file .exe, 
        // bấm trong lúc chạy thử ở Unity Editor nó sẽ không tự tắt đâu nhé.
    }
}