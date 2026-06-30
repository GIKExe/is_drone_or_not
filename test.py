from ultralytics import YOLO


def main():
    model = YOLO("best.pt")

    while True:
        img_path = input("Введите путь к фото → ").strip().strip("'").strip('"')

        results = model(img_path, save=False, verbose=False)[0]

        q = float(results.probs.data[0])
        nq = float(results.probs.data[1])
        print(f'Дрон: {q:.3f}, НЕ дрон: {nq:.3f}, Сумма: {nq+q:.3f}')


if __name__ == '__main__':
    try:
       main()
    except KeyboardInterrupt:
        pass
    except:
        raise
