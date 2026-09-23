import time
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import open3d as o3d
from PIL import Image, ImageFilter

# максимальный размер картинки, чтобы всё не тормозило
MAX_SIZE = 800
# берём не каждый пиксель, а через шаг, иначе точек будет очень много
STEP = 2


def load_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    return img


def resize_image(img):
    w, h = img.size
    if w > MAX_SIZE or h > MAX_SIZE:
        if w > h:
            new_w = MAX_SIZE
            new_h = int(h * MAX_SIZE / w)
        else:
            new_h = MAX_SIZE
            new_w = int(w * MAX_SIZE / h)
        img = img.resize((new_w, new_h))
    return img


def get_depth(img):
    # переводим в numpy массив чтобы было проще считать
    arr = np.array(img).astype(np.float32)

    # переводим в оттенки серого (яркость)
    gray = arr[:, :, 0] * 0.3 + arr[:, :, 1] * 0.6 + arr[:, :, 2] * 0.1
    gray = gray / 255

    # немного размываем, чтобы найти контраст
    blur_img = img.filter(ImageFilter.GaussianBlur(2))
    blur_arr = np.array(blur_img).astype(np.float32)
    blur_gray = blur_arr[:, :, 0] * 0.3 + blur_arr[:, :, 1] * 0.6 + blur_arr[:, :, 2] * 0.1
    blur_gray = blur_gray / 255

    contrast = abs(gray - blur_gray)
    if contrast.max() > 0:
        contrast = contrast / contrast.max()

    # ищем границы через векторные операции NumPy (вместо медленного
    # попиксельного Python-цикла — см. PROFILING.md, ускорение в разы)
    h, w = gray.shape
    edges = np.zeros((h, w), dtype=np.float32)

    dx = np.abs(gray[1:, 1:] - gray[1:, :-1])
    dy = np.abs(gray[1:, 1:] - gray[:-1, 1:])

    edges[1:, 1:] = dx + dy
    edges = np.clip(edges, 0, 1)

    # итоговая глубина - чем темнее пиксель тем он как бы ближе
    depth = (1 - gray) * 0.65 + edges * 0.2 + contrast * 0.15
    depth = np.clip(depth, 0.01, 1)

    return depth


def save_depth_image(path, depth):
    img_data = (depth * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save(path)


def make_point_cloud(img, depth):
    colors_arr = np.array(img).astype(np.float32) / 255
    h, w = depth.shape

    focal = max(w, h)
    cx = w / 2
    cy = h / 2

    # берём точки с шагом STEP через срезы NumPy вместо Python-циклов
    # с append (это ещё и убирает лишний рост списков в памяти)
    ys = np.arange(0, h, STEP)
    xs = np.arange(0, w, STEP)

    jj, ii = np.meshgrid(xs, ys)

    z = depth[ii, jj]
    x = (jj - cx) * z / focal
    y = (ii - cy) * z / focal

    points = np.stack([x, -y, z], axis=-1).reshape(-1, 3)
    colors = colors_arr[ii, jj].reshape(-1, 3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def process():
    filename = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Картинки", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Все файлы", "*.*")]
    )

    if filename == "":
        return

    try:
        img = load_image(filename)
        img = resize_image(img)

        # замер времени выполнения для профилирования (см. PROFILING.md)
        start_time = time.time()
        depth = get_depth(img)
        elapsed = time.time() - start_time
        print(f"[PROFILING] get_depth выполнилась за {elapsed:.3f} сек")

        # обрезаем расширение файла и делаем новые имена
        name_without_ext = filename.rsplit(".", 1)[0]
        depth_path = name_without_ext + "_depth.png"
        cloud_path = name_without_ext + "_point_cloud.ply"

        save_depth_image(depth_path, depth)

        cloud = make_point_cloud(img, depth)

        o3d.io.write_point_cloud(cloud_path, cloud, write_ascii=True)

        messagebox.showinfo("Готово", "Файлы сохранены:\n" + depth_path + "\n" + cloud_path)

        o3d.visualization.draw_geometries([cloud])

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


# основное окно программы
window = tk.Tk()
window.title("3D облако точек")
window.geometry("500x250")

label1 = tk.Label(window, text="Создание 3D облака точек", font=("Arial", 16))
label1.pack(pady=20)

label2 = tk.Label(window, text="Загрузите картинку (PNG, JPG, BMP, TIFF, WebP)")
label2.pack(pady=5)

btn = tk.Button(window, text="Выбрать изображение", command=process, width=25, height=2)
btn.pack(pady=20)

window.mainloop()