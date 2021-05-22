# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-22T13:30:01+02:00
# @License: MIT



import networkx as nx
import scipy as sc
from scipy.spatial import Voronoi
import random as rd
import sys
import kirchhoff_init

class random_graph(kirchhoff_init.circuit):

    avg_dist=[]
    def mirror_boxpoints(self,points,sl):

        points_matrix=points
        intervall=[-1,0,1]
        for i in intervall:
            for j in intervall:
                if (i!=0 or j!=0):
                    points_matrix=sc.concatenate((points_matrix,points+(i*sl,j*sl)))

        return points_matrix
    def mirror_cubepoints(self,points,sl):

        points_matrix=points
        intervall=[-1,0,1]
        for i in intervall:
            for j in intervall:
                for k in intervall:
                    if (i!=0 or j!=0 or k!=0):
                        points_matrix=sc.concatenate((points_matrix,points+(i*sl,j*sl,k*sl)))

        return points_matrix
    # construct random 3d graph, confined in a box
    def construct_voronoi_periodic(self,number,sidelength,mode):

        V=0
        # create points for voronoi tesselation
        if mode=='2D':
            XY=[]

            for i in range(number):
                x=rd.uniform(0,sidelength)
                y=rd.uniform(0,sidelength)

                XY.append((x,y))
            self.XY=XY
            XY=self.mirror_boxpoints(sc.array(XY),sidelength)
            self.XY_periodic=XY
            V=Voronoi(XY)

        if mode=='3D':
            XYZ=[]

            for i in range(number):
                x=rd.uniform(0,sidelength)
                y=rd.uniform(0,sidelength)
                z=rd.uniform(0,sidelength)

                XYZ.append((x,y,z))
            self.XYZ=XYZ
            XYZ=self.mirror_cubepoints(sc.array(XYZ),sidelength)
            self.XYZ_periodic=XYZ
            V=Voronoi(XYZ)

        return V

    def random_voronoi_periodic(self,number,sidelength,mode,pipe_length,conductance,flow):

        self.k=conductance
        self.l=pipe_length
        self.f=flow
        #construct a core of reandom points in 2D box for voronoi tesselation, mirror the primary box so a total of 9 cells is created with the initial as core
        V=self.construct_voronoi_periodic(number,sidelength,mode)
        #pick up the face of the core which correspond to a periodic voronoi lattice
        faces=[]
        for j,i in enumerate(V.point_region):
            faces.append(sc.asarray(V.regions[i]))

            if j==number-1:
                break
        #use periodic kernel to construct the correponding network
        faces=sc.asarray(faces)
        f=faces[0]


        for i in range(len(faces[:])):
            if i+1==len(faces[:]):
                break
            f=sc.concatenate((f,faces[i+1]))
        for i in faces:
            for j in i:
                v=V.vertices[j]
                self.G.add_node(j,pos=v,lablel=j)
        if mode=='2D':

            k=0
            for i in V.ridge_vertices:

                mask=sc.in1d(i,f)
                if sc.all( mask == True ):

                    for l in range(len(i)):
                        h=len(i)-1
                        self.G.add_edge(i[h-(l+1)],i[h-l],slope=(V.vertices[i[h-(l+1)]],V.vertices[i[h-l]]), label=k)
                        k+=1
                        if len(i)==2:
                            break
        if mode=='3D':

            k=0
            for i in V.ridge_vertices:

                mask=sc.in1d(i,f)
                if sc.all( mask == True ):

                    for l in range(len(i)):
                        h=len(i)-1
                        self.G.add_edge(i[h-(l+1)],i[h-l],slope=(V.vertices[i[h-(l+1)]],V.vertices[i[h-l]]), label=k)
                        k+=1
                        if len(i)==2:
                            break
        # initialze circuit & add attributes
        self.initialize_circuit()

    def is_in_box(self,v,sl):
        answer=True

        if (v[0] > sl) or (v[0] < -sl):
            answer=False
        if (v[1] > sl) or (v[1] < -sl):
            answer=False
        if (v[2] > sl) or (v[2] < -sl):
            answer=False

        return answer
  # construct random 3d graph, confined in a certain spherical boundary, connections set via voronoi tesselation
    def random_voronoi_tesselation(self,number,sidelength,pipe_length,conductance,flow):

        self.k=conductance
        self.l=pipe_length
        self.f=flow

        # coordinates entry point
        core_x=0
        core_y=0
        core_z=0

        XYZ=[(core_x,core_y,core_z)]
        # create points for voronoi tesselation
        j=0
        while True:
            x=rd.uniform(-sidelength,sidelength)
            y=rd.uniform(-sidelength,sidelength)
            z=rd.uniform(-sidelength,sidelength)

            XYZ.append((x,y,z))
            j+=1
            if j==number:
                break

        V=Voronoi(XYZ)
        j=0
        control=[]
        for i in V.vertices:
            if self.is_in_box(i,sidelength):
                self.G.add_node(j,pos=i, label=j)
                # print('node:'+str(j))
            j+=1

        k=0
        for i in V.ridge_vertices:

                for j in range(len(i)):
                    h=len(i)-1
                    if self.is_in_box(V.vertices[i[h-(j+1)]],sidelength) and self.is_in_box(V.vertices[i[h-j]],sidelength):
                        if i[h-(j+1)]!=-1 and i[h-j]!=-1:
                            self.G.add_edge(i[h-(j+1)],i[h-j],slope=(V.vertices[i[h-(j+1)]],V.vertices[i[h-j]]), label=k)
                            # print('edge1:'+str(i[h-(j+1)])+' edge2:'+str(i[h-(j)]))

                        k+=1
            # initialze circuit & add attributes
        # print(self.G.nodes())
        self.initialize_circuit()

    def random_voronoi_sphere(self,number,sidelength,pipe_length,conductance,flow):
        self.k=conductance
        self.l=pipe_length
        self.f=flow
        # coordinates entry point
        core_x=0
        core_y=0
        core_z=0

        XYZ=[(core_x,core_y,core_z)]
        # create points for voronoi tesselation
        j=0
        while True:
            x=rd.uniform(-sidelength,sidelength)
            y=rd.uniform(-sidelength,sidelength)
            z=rd.uniform(-sidelength,sidelength)
            if x*x+y*y+z*z <= sidelength*sidelength:
                XYZ.append((x,y,z))
                j+=1
            if j==number:
                break

        V=Voronoi(XYZ)
        for j,i in enumerate(V.points):
            self.G.add_node(j,pos=i, label=j)

        j=0
        for i,k in zip(V.ridge_points,V.ridge_vertices):
            k=sc.asarray(k)
            if sc.all(k >= 0):

                self.G.add_edge(i[0],i[1],slope=(V.points[i[0]],V.points[i[1]]), label=j)

                j+=1
            # initialze circuit & add attributes
        self.initialize_circuit()

    def random_voronoi_cube(self,number,sidelength,pipe_length,conductance,flow):

        self.k=conductance
        self.l=pipe_length
        self.f=flow
        # coordinates entry point
        core_x=0
        core_y=0
        core_z=0

        XYZ=[(core_x,core_y,core_z)]

        # create points for voronoi tesselation
        for i in range(number):

            x=rd.uniform(-sidelength,sidelength)
            y=rd.uniform(-sidelength,sidelength)
            z=rd.uniform(-sidelength,sidelength)

            XYZ.append((x,y,z))

        V=Voronoi(XYZ)
        for j,i in enumerate(V.points):
            self.G.add_node(j,pos=i, label=j)

        j=0
        for i,k in zip(V.ridge_points,V.ridge_vertices):
            k=sc.asarray(k)
            if sc.all(k >= 0):

                self.G.add_edge(i[0],i[1],slope=(V.points[i[0]],V.points[i[1]]), label=j)

                j+=1

        # initialze circuit & add attributes
        self.initialize_circuit()
        V.close()

    def random_voronoi_square(self,sidelength,pipe_length,conductance,flow):

        self.k=conductance
        self.l=pipe_length
        self.f=flow
        # coordinates entry point
        core_x=0
        core_y=0

        XY=[(core_x,core_y)]
        n=int((sidelength**2)/2)

        # create points for voronoi tesselation
        for i in range(n):

            x=rd.uniform(-sidelength,sidelength)
            y=rd.uniform(-sidelength,sidelength)

            XY.append((x,y))

        V=Voronoi(XY)

        for j,i in enumerate(V.points):
            self.G.add_node(j,pos=i, label=j)

        j=0
        for i,k in zip(V.ridge_points,V.ridge_vertices):
            k=sc.asarray(k)
            if sc.all(k >= 0):

                self.G.add_edge(i[0],i[1],slope=(V.points[i[0]],V.points[i[1]]), label=(i[0],i[1]))
                j+=1


        self.initialize_circuit()
        V.close()

    def random_voronoi_circle(self,number,sidelength,pipe_length,conductance,flow):
        self.k=conductance
        self.l=pipe_length
        self.f=flow
        # coordinates entry point
        core_x=0
        core_y=0

        XY=[(core_x,core_y)]

        # create points for voronoi tesselation
        j=0
        while True:
            x=rd.uniform(-sidelength,sidelength)
            y=rd.uniform(-sidelength,sidelength)

            if x*x+y*y <= sidelength*sidelength:
                XY.append((x,y))
                j+=1
            if j==number:
                break

        V=Voronoi(XY)
        H=nx.DiGraph()
        for j,i in enumerate(V.points):
            H.add_node(j,pos=i, label=j)

        j=0

        for i,k in zip(V.ridge_points,V.ridge_vertices):
            k=sc.asarray(k)
            if sc.all(k >= 0):
                H.add_edge(i[0],i[1],slope=(V.points[i[0]],V.points[i[1]]), label=(i[0],i[1]))
                j+=1

        self.G=H
        self.initialize_circuit()
