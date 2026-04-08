using UnityEngine;
using UnityEngine.UI;
using TMPro; // Đừng quên thư viện này để dùng TextMeshPro

public class InventoryManager : MonoBehaviour
{
    [Header("Giao diện Tủ Đồ")]
    public GameObject inventoryPanel; 
    public Transform gridLayout;      // Kéo cái "ItemGrid" (Lưới) vào đây
    public GameObject itemPrefab;     // Kéo cái Khuôn đúc 1 ô vật phẩm từ Project vào đây

    // --- CÁC HÀM UI CƠ BẢN ---
    public void OpenInventory()
    {
        if (inventoryPanel != null) inventoryPanel.SetActive(true);
    }

    public void CloseInventory()
    {
        if (inventoryPanel != null) inventoryPanel.SetActive(false);
    }

    // --- HÀM LOGIC XỬ LÝ VẬT PHẨM ---
    
    // Python API (Object Detection) sau khi quét xong sẽ gọi hàm này
    public void AddNewItem(string itemName, Sprite itemIcon = null)
    {
        // 1. Dập khuôn tạo ra 1 ô đồ mới, tự động nhét nó làm con của ItemGrid
        GameObject newItem = Instantiate(itemPrefab, gridLayout);

        // 2. Tìm Object con có tên "ItemIcon" để thay hình (nếu có)
        if (itemIcon != null)
        {
            Image iconImage = newItem.transform.Find("ItemIcon").GetComponent<Image>();
            if (iconImage != null) iconImage.sprite = itemIcon;
        }

        // 3. Tìm Object con có tên "ItemName" để thay chữ
        TextMeshProUGUI nameText = newItem.transform.Find("ItemName").GetComponent<TextMeshProUGUI>();
        if (nameText != null) nameText.text = itemName;

        Debug.Log($"[Inventory] Píp! Đã quét và cất [{itemName}] vào tủ đồ!");
    }

    // Nút Nháp: Gọi hàm này để test UI trước khi có Python
    public void TestAddItem()
    {
        AddNewItem("Món đồ Sinh Viên " + Random.Range(1, 100)); 
    }
}