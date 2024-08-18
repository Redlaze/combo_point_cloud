import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import copy
import numpy as np
import open3d as o3d
import csv

feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")


MU, SIGMA = 0, 0.00003
THRESHOLD = 0.5

# фиксирование параметров камеры
CAM_PARAMS = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)


def get_depth(path):
    # load and resize the input image
    image = Image.open(f"test/{path}.jpg")
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # remove borders
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    return image, output


def draw_reg_result(source, target, transformation, writer=False):
    """
    Функция комбинирования облаков точек
    :param source: исходное облако
    :param target: целевое облако
    :param transformation: матрица трансформации
    :param writer: флаг записи файла
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    if writer:
        o3d.io.write_point_cloud('cloud.pcd', source_temp + target_temp)


def apply_noise(pcd, mu, sigma):
    """
    Функция создания шумового облака точек
    :param pcd: исходное облако
    :param mu: среднее
    :param sigma: СКО
    :return: шумовое облако точек
    """
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def get_point_cloud():
    """
    Функция создания комбинаций облаков точек
    и расчета траектории камеры
    """
    trajectory = []
    for i in range(1, 4):
        print(i)

        # создание карт глубины
        source_color, source_depth = get_depth(i)
        target_color, target_depth = get_depth(i + 1)

        source_color = np.array(source_color)
        target_color = np.array(target_color)

        source_depth = (source_depth * 255 / np.max(source_depth)).astype('uint8')
        target_depth = (target_depth * 255 / np.max(target_depth)).astype('uint8')

        source_depth = o3d.geometry.Image(source_depth)
        source_color = o3d.geometry.Image(source_color)

        target_depth = o3d.geometry.Image(target_depth)
        target_color = o3d.geometry.Image(target_color)

        # создание отдельных облаков точек
        source_rgb_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth,
                                                                              convert_rgb_to_intensity=False)
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgb_image, CAM_PARAMS)

        target_rgb_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth,
                                                                              convert_rgb_to_intensity=False)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgb_image, CAM_PARAMS)

        # вычисление одометрии
        option = o3d.pipelines.odometry.OdometryOption()
        odo_init = np.identity(4)

        [success_color_term, trans_color_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgb_image,
            target_rgb_image,
            CAM_PARAMS,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            option)

        trajectory.append(trans_color_term)

        # создание шумового поля
        source_noisy = apply_noise(source_pcd, MU, SIGMA)

        # загрузка предыдущего облака точек
        if i > 1:
            source_pcd = o3d.io.read_point_cloud('cloud.pcd')

        # поиск оптимального соединения облаков через Robust Kernels
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        loss = o3d.pipelines.registration.TukeyLoss(k=SIGMA)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target_pcd, THRESHOLD, trans_color_term, p2l)
        draw_reg_result(source_pcd, target_pcd, reg_p2l.transformation, writer=True)

    # сохранение файла траектории камеры
    trajectory = np.asarray(trajectory)
    with open('trajectoty.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(trajectory)

    # сохранение облака точек в формате *.ply
    pcd = o3d.io.read_point_cloud('cloud.pcd')
    o3d.io.write_point_cloud('result.ply', pcd)