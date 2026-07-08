from ultralytics import YOLO
from spectrogram import IMG_SIZE


def main():
    model = YOLO('yolov8n-cls.pt')

    model.train(
        data='out',
        epochs=20,
        imgsz=IMG_SIZE,
        batch=1024,
        workers=4,
        device=0,
        cache=False,
        val=True,

        optimizer='auto',
        amp=True,
        rect=True,

        mask_ratio=1,
        degrees=0.0,
        angle=0.0,
        mosaic=0.0,
        erasing=0.0,
        mixup=0.0,
        cutmix=0.0,
        fliplr=0.0,
        flipud=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        auto_augment=None,
        augment=False,
        close_mosaic=10,

        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        patience=100,
        seed=0,
        deterministic=True,
        single_cls=False,
        overlap_mask=True,
        nbs=64,
        dropout=0.0,
        freeze=None,
        multi_scale=0.0,
        fraction=1.0,
        copy_paste=0.0,
        copy_paste_mode='flip',

        plots=True,
        verbose=True,
        exist_ok=False,
        project=None,
        compile=False,
    )

    print("Обучение успешно завершено!")


if __name__ == '__main__':
    main()