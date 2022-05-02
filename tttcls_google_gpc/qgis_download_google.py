# -*- coding: utf-8 -*-
import os
from osgeo import ogr

# positive sample: GPC 
#shp_path = r'.\GPC\GPC_rect_prj.shp' # annotated image-level GPC sample points, projection: EPSG:3857
#img_path = r'.\GPC' # save path
# negative sample: non-GPC
shp_path = r'.\negative\negative_512_30k_rect_rpj.shp' # 30k negative sample points
img_path = r'.\negative' # save path

img_suffix = 'png'

# pixel size: 1 meters
# px_geosize = [0.5, 0.5]
px_geosize = [1, 1]
# padding size for the bounding box
#pad_size = [0, 0, 0, 0]

pad_geosize = [0,0,0,0]
# pad_geosize = [a * b for a, b in zip(pad_size, px_geosize)]

shp = ogr.Open(shp_path)
lyr = shp.GetLayer()
for feat in lyr:
    # default setting: shapfile has the "id" field. If not exist, define it using the id number
    name = str(feat.GetField('id'))
    respath = os.path.join(img_path, name + '.' + img_suffix)
    if os.path.exists(respath):
        continue
    geom = feat.GetGeometryRef()
    geom_ext = geom.GetEnvelope()
    
    # the geolocation of cropped images
    clip_ext = [geom_ext[0] - pad_geosize[0], 
    geom_ext[2] - pad_geosize[1],
    geom_ext[1] + pad_geosize[2], 
    geom_ext[3] + pad_geosize[3]]
    # print(pad_geosize)
    # the pixel size of cropped images
    clip_size = [int((clip_ext[2] - clip_ext[0]) / px_geosize[0]), int((clip_ext[3] - clip_ext[1]) / px_geosize[1])]

    # change to the qgis form
    rect = QgsRectangle(clip_ext[0], clip_ext[1], clip_ext[2], clip_ext[3])

    # settings for saving images
    settings = iface.mapCanvas().mapSettings()

    # set the total size for images
    settings.setExtent(rect)
    
    # set the pixel resolution for images
    settings.setOutputSize(QSize(clip_size[0], clip_size[1]))
    job = QgsMapRendererSequentialJob(settings)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    image.save(os.path.join(img_path, name + '.' + img_suffix))

print('task finished!')
