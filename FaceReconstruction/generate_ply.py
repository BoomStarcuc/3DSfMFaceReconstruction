
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from imageio import imread,imwrite
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    
    imgpath = '/cis/otherstu/yw2009/anaconda3/envs/Vo_face/SfMFaceBasedonMassiveLandmark/FaceReconstruction/test_image/18_eye_closed_0009_090.png'
    disppath = '/cis/otherstu/yw2009/anaconda3/envs/Vo_face/SfMFaceBasedonMassiveLandmark/FaceReconstruction/output/-18_eye_closed_0009_090_disp.png'
    
    imgL = cv.imread(imgpath)[:,:,:3]
    imgL = cv.resize(imgL,(384,384))

    disp = cv.imread(disppath,0)
    disp_forsave = cv.imread(disppath)
    
    
    color_mask = (imgL < imgL.max())[:,:,0]
    
    disp = disp*color_mask
    
    disp_forsave[color_mask == 0] = 0
    cv.imwrite(disppath,disp_forsave)
    
    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 1.5* w                        # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    
    
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    
    
    out_fn = disppath[:-4]+'.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()