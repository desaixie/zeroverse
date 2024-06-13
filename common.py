
import re
import numpy as np
import  sys
import os





def Det(v1, v2, v3):
    det = v1[0] * v2[1] * v3[2] + \
          v2[0] * v3[1] * v1[2] + \
          v3[0] * v1[1] * v2[2] - \
          v3[0] * v2[1] * v1[2] - \
          v2[0] * v1[1] * v3[2] - \
          v1[0] * v3[1] * v2[2]

    return det

def triangleIntersect(start, direction, ta, tb, tc):
    EPS = 0.000001

    e1 = np.reshape(ta, 3) - np.reshape(tb, 3)
    e2 = np.reshape(ta, 3) - np.reshape(tc, 3)

    s = np.reshape(ta,3) - np.reshape(start,3)

    det = Det(direction, e1, e2)
    t = Det(s, e1, e2) / det
    be = Det(direction, s, e2) / det
    ga = Det(direction, e1, s) / det

    if (t > EPS and be >=0 and ga >= 0 and be + ga <= 1):
        return True

    return False

def isoNormalMat(targetVec):
    targetZ = normalize(targetVec).reshape(-1)
    candY = (0,1,0)
    if targetZ.dot(candY) > 0.99:
        candY = (1,0,0)
    targetX = normalize(np.cross(candY, targetZ)).reshape(-1)
    targetY = np.cross(targetZ, targetX)

    mat = np.row_stack([targetX, targetY, targetZ])
    return mat

def VecToEnvDir(vecs):
    targetMat = np.row_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rads = VecToSph(targetMat.dot(np.reshape(vecs,(-1,3)).T).T)
    phis = rads[:,0]

    phis[phis<np.pi/2.0] += np.pi*2.0

    us = (phis - np.pi/2.0) / (np.pi*2.0)
    vs = (rads[:,1]) / np.pi

    return np.column_stack([us,vs])




def createMaskedSG(color=(1.0,1.0,1.0), mu = 1.0, axis = (0,0,1), maskAxis = (0,0,1), maskDeg = 180, res = (512, 256)):
    initShape = res
    uvMap = np.ones((initShape[1], initShape[0], 2))

    # the environment maps faces have fixed orientations regarding the scene:
    # env up, scene up; env mid right, scene right; env center, scene forward
    # the order the directions appear on the pixel should match the target relighting axes.

    # original phi theta: theta :0~pi pi:1.5pi ~ -0.5pi, this is the original order,
    # dir transforms should make the order match the target relighting axes
    uvMap[:, :, 1] = np.linspace(-1, 1, initShape[1]).reshape(-1, 1) * (
            np.pi * 0.5 - np.pi / (initShape[1]) * 0.5) + np.pi * 0.5
    uvMap[:, :, 0] = np.linspace(1, -1, initShape[0]).reshape(1, -1) * (
        np.pi - 2.0 * np.pi / (initShape[0]) * 0.5) + np.pi * 0.5
    uvMap = uvMap.reshape((-1, 2))

    #angles to dirs,
    orgDirs = SphToVec(uvMap)

    # transform the dirs in two steps so that the dir in each pixel represents the dir in a mitsuba environment map.
    # in fact the two steps is not that intuitive and can be replaced by one step,
    # simply making the sampling axes compatible with the relighting axes [[1,0,0], [0,0,1], [0,-1, 0]]


    #transform dirs into dirs in mitsuba
    targetMat = np.row_stack([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    relightDirs = targetMat.dot(orgDirs.T)
    axis = normalize(axis)
    sgImg =  np.exp(mu * (relightDirs.T.dot(axis) - 1.0))
    integral = 2.0 * np.pi / mu * (1.0 - np.exp(-2.0 * mu))
    sgImg = sgImg / integral

    mask = relightDirs.T.dot(maskAxis)
    sgImg[mask < np.cos(np.deg2rad(maskDeg))] = 0

    return np.reshape(color, (3)) * sgImg.reshape((res[1], res[0],1))


def loadVec(filePath, interC = " ", typeV = float):
    vecs = []
    with open(filePath, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            if line[-1] == "\n":
                line = line[:-1]

            dir = [typeV(x) for x in line.split(interC)]
            vecs.append(dir)
    return vecs


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def rotateVector(vector, axis, angle):
    cos_ang = np.reshape(np.cos(angle),(-1));
    sin_ang = np.reshape(np.sin(angle),(-1));
    vector = np.reshape(vector,(-1,3))
    axis = np.reshape(np.array(axis),(-1,3))
    return vector * cos_ang[:,np.newaxis] + axis*np.dot(vector,np.transpose(axis))*(1-cos_ang)[:,np.newaxis] + np.cross(axis,vector) * sin_ang[:,np.newaxis]

def normalize(x):
    if(len(np.shape(x)) == 1):
        return x/np.linalg.norm(x)
    else:
        return x/np.linalg.norm(x,axis=1)[:,np.newaxis]

#return Phi(0, 2pi), Theta (0, pi) in rads
def VecToSph(coords):

    coords = np.reshape(coords,(-1,3))

    rads = np.zeros((coords.shape[0],2))

    rads[:,0] = np.arctan2(coords[:,1], coords[:,0])
    rads[rads<0] += 2.0 * np.pi
    rads[:,1] = np.arccos(coords[:,2])

    return rads

#Phi Theta
def SphToVec(coords):
    coords = np.reshape(coords,(-1,2))

    vec = np.zeros((coords.shape[0],3))
    vec[:,0] = np.cos(coords[:,0])*np.sin(coords[:,1])
    vec[:,1] = np.sin(coords[:,0])*np.sin(coords[:,1])
    vec[:,2] = np.cos(coords[:,1])
    return vec


def subPixels(img, xs, ys, bg=0):
    height = img.shape[0]
    width = img.shape[1]
    xs = np.reshape(xs, -1)
    ys = np.reshape(ys, -1)
    ix0 = xs.astype(int)
    iy0 = ys.astype(int)
    ix1 = ix0+1
    iy1 = iy0+1
    badIds = []
    ids = np.reshape(np.where(ix0 < 0), -1)
    badIds = np.append(badIds, ids)
    if len(ids) > 0:
        ix0[ids]=0
        ix1[ids]=0
    ids = np.reshape(np.where(iy0 < 0), -1)
    badIds = np.append(badIds, ids)
    if len(ids) > 0:
        iy0[ids] = 0
        iy1[ids] = 0
    ids = np.reshape(np.where(ix1 > width-1), -1)
    badIds = np.append(badIds, ids)
    if len(ids) > 0:
        ix0[ids] = width-1
        ix1[ids] = width-1
    ids = np.reshape(np.where(iy1 > height - 1), -1)
    badIds = np.append(badIds, ids)
    if len(ids) > 0:
        iy0[ids] = height - 1
        iy1[ids] = height - 1


    ratex = xs - ix0
    ratey = ys - iy0
    if len(img.shape) > 2:
        ratex = ratex.reshape((-1,1))
        ratey = ratey.reshape((-1, 1))

    px0_y0 = img[(iy0,ix0)]
    px0_y1 = img[(iy1,ix0)]
    px1_y0 = img[(iy0,ix1)]
    px1_y1 = img[(iy1,ix1)]

    py0 = px0_y0 * (1.0-ratex) + px1_y0*ratex
    py1 = px0_y1 * (1.0-ratex) + px1_y1*ratex
    p = py0 * (1 - ratey) + py1 * ratey

    badIds = badIds.astype(int)
    if len(badIds) > 0:
        p[badIds] = bg
    return p


def subPix(img, x, y):
    height = img.shape[0]
    width = img.shape[1]
    ix0 = int(x)
    iy0 = int(y)
    ix1 = ix0+1
    iy1 = iy0+1

    if ix0 < 0:
        ix0=ix1=0
    if iy0 < 0:
        iy0=iy1=0
    if ix1 > width-1:
        ix0=ix1=width-1
    if iy1 > height-1:
        iy0=iy1=height-1

    ratex = x - ix0
    ratey = y - iy0

    px0_y0 = img[iy0,ix0]
    px0_y1 = img[iy1,ix0]
    px1_y0 = img[iy0,ix1]
    px1_y1 = img[iy1,ix1]

    py0 = px0_y0 * (1.0-ratex) + px1_y0*ratex
    py1 = px0_y1 * (1.0-ratex) + px1_y1*ratex
    p = py0 * (1 - ratey) + py1 * ratey
    return p





def load_pfm(filename):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    file = open(filename,'rb')

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)

    reverseIds = range(len(data)-1, -1, -1)

    out = data[reverseIds]


    return out

'''
Save a Numpy array to a PFM file.
'''

def save_pfm(filename, image, scale = 1):

    file = open(filename,'wb')

    color = None
    image = image.astype(np.float32)
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    reverseIds = range(len(image)-1, -1, -1)

    out = image[reverseIds]


    out.tofile(file)

def saveAsPly(filename, points, color = (255, 0, 0)):
    color = np.reshape(color, (-1,3))
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n"%(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n end_header")
        for i,point in enumerate(points):
            if len(color) != len(points):
                f.write("\n%.5f %.5f %.5f %d %d %d"%(point[0], point[1], point[2], color[0][0], color[0][1], color[0][2]))
            else:
                f.write("\n%.5f %.5f %.5f %d %d %d"%(point[0], point[1], point[2], color[i][0], color[i][1], color[i][2]))

def make_non_exist_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def saveToPly(filename,verts,faces=None,norms=None,colors=None):
    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    if verts is not None:
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
    if norms is not None:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    if colors is not None:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    
    if faces is not None:
        ply_file.write("element face %d\n"%(faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        if verts is not None:
            ply_file.write("%f %f %f "%(verts[i,0],verts[i,1],verts[i,2]))
        if norms is not None:
            ply_file.write("%f %f %f "%(norms[i,0],norms[i,1],norms[i,2]))
        if colors is not None:
            ply_file.write("%d %d %d "%(colors[i,0],colors[i,1],colors[i,2]))

        ply_file.write('\n')
    
    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()




def saveToPlyWithBrdf(filename,verts,faces=None,norms=None,colors=None,albedo=None,
        roughness=None):
    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    if verts is not None:
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
    if norms is not None:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    if colors is not None:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    if albedo is not None:
        ply_file.write("property albedo uchar\n")
        ply_file.write("property albedo uchar\n")
        ply_file.write("property albedo uchar\n")

    if roughness is not None:
        ply_file.write("property roughness uchar\n")
        ply_file.write("property roughness uchar\n")
        ply_file.write("property roughness uchar\n")

    
    if faces is not None:
        ply_file.write("element face %d\n"%(faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        if verts is not None:
            ply_file.write("%f %f %f "%(verts[i,0],verts[i,1],verts[i,2]))
        if norms is not None:
            ply_file.write("%f %f %f "%(norms[i,0],norms[i,1],norms[i,2]))
        if colors is not None:
            ply_file.write("%d %d %d "%(colors[i,0],colors[i,1],colors[i,2]))

        if albedo is not None:
            ply_file.write("%d %d %d "%(albedo[i,0], albedo[i,1], albedo[i,2]))

        if roughness is not None:
            ply_file.write("%d %d %d "%(roughness[i,0],roughness[i,1],roughness[i,2]))
        
        ply_file.write('\n')
    
    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()




