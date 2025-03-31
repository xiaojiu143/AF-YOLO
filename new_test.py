from ultralytics import YOLO
from ultralytics.utils import ASSETS

CFG = "ultralytics/cfg/models/Mine_two/yDDn.yaml" ########
SOURCE = ASSETS / "bus.jpg"


def test_model_forward():
    """Test the forward pass of the model. """
    try:
        print("Initializing YOLO model...")
        model = YOLO(CFG)
        print("Model initialized successfully.")

        print("Running forward pass...")
        result = model(source=None, imgsz=640, augment=True)
        print("Forward pass completed successfully.")
        print("Result:", result)

        # 如果执行到这里没有异常，表示测试成功
        print("✅ 测试成功！")
    except Exception as e:
        print("❌ 测试失败，错误信息如下：", e)  # 捕获异常并打印


# 执行测试函数
if __name__ == "__main__":
    test_model_forward()