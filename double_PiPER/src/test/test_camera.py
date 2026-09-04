import cv2

# カメラを開く（0はデフォルトカメラ）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けません")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("フレームを取得できません")
        break

    # 画像を表示
    cv2.imshow("Web Camera", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()