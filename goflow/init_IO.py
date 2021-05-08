import time
import numpy as np
import scipy.linalg as lina
import os, errno
import os.path as op
import shutil
import datetime
import networkx as nx
import glob

class output_organizer:

    def __init__(self):
        self.DIR_OUT = '../../../output/'
        self.DIR_OUT_PLOT=''
        self.DIR_OUT_DATA=''

        self.DIR_OUT_PROGRAM=''
        self.DIR_OUT_PROGRAM_M=''
        self.DIR_SIM=''
        self.DIR_OUT_PROGRAM_PLOT=''
        self.DIR_OUT_PROGRAM_DATA=''
        self.BACKUP=''

    def create_dir(self,directory):

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def create_output_paths(self,program):
        self.DIR_OUT_PROGRAM=op.join( self.DIR_OUT , program )
        self.DIR_OUT_DATA=op.join( self.DIR_SIM , 'data' )
        self.DIR_OUT_PLOT=op.join( self.DIR_SIM , 'eval' )

        self.create_dir(self.DIR_OUT_PROGRAM)
        self.create_dir(self.DIR_SIM)
        self.create_dir(self.DIR_OUT_DATA)
        self.create_dir(self.DIR_OUT_PLOT)

    def create_output_directories(self):
        self.create_dir(op.join(self.DIR_OUT_PROGRAM , self.DIR_OUT_PROGRAM_M))
        self.create_dir(self.DIR_OUT_PROGRAM_PLOT)
        self.create_dir(self.DIR_OUT_PROGRAM_DATA)

    def init_output_directory(self,program):

        DT=datetime.datetime.now()

        self.DIR_OUT_PROGRAM=op.join(self.DIR_OUT, program )
        self.DIR_OUT_PROGRAM_M=op.join(self.DIR_OUT_PROGRAM_M,DT.strftime('%Y%m%d_%H%M'))
        self.DIR_OUT_PROGRAM_PLOT=op.join(self.DIR_OUT_PROGRAM , self.DIR_OUT_PROGRAM_M)
        self.DIR_OUT_PROGRAM_DATA=op.join(self.DIR_OUT_PROGRAM , self.DIR_OUT_PROGRAM_M)

        self.create_output_directories()

    def init_output_path(self,program,comment):

        self.DIR_SIM=op.join( self.DIR_OUT_PROGRAM , self.DIR_OUT_PROGRAM_M + comment)
        self.BACKUP=self.DIR_SIM
        self.create_output_paths(program)

    def init_code_backup(self,loc_path,program):
        dst = os.path.join(self.BACKUP,'source_code')
        if not op.exists(dst):
            try:
                wildcard_expression = op.join(loc_path,"*py")
                src = glob.glob(wildcard_expression)
                src.append(op.join(os.getcwd(),program))

                os.makedirs(dst)
                for f in src:
                    shutil.copy(f,dst)

            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def save_nparray(self, nparray,label):

        np.save(op.join(self.DIR_OUT_DATA,label),nparray)

class output_organizer_kirchhoff(output_organizer,object):

    def __init__(self):
        super(output_organizer_kirchhoff,self).__init__()
        self.MEASURE_SETTING=[]
        self.DIR_SETTING='../../setting_py/'

    def init_output_directory(self,program):

        DT=datetime.datetime.now()
        self.DIR_OUT_PROGRAM=op.join(self.DIR_OUT , program)
        self.DIR_OUT_PROGRAM_M=op.join(self.DIR_OUT_PROGRAM_M,  DT.strftime('%Y%m%d_%H%M'))
        self.DIR_OUT_PROGRAM_PLOT=op.join( self.DIR_OUT_PROGRAM , op.join(self.DIR_OUT_PROGRAM_M , 'sweep_eval'))
        self.DIR_OUT_PROGRAM_DATA=op.join( self.DIR_OUT_PROGRAM , op.join(self.DIR_OUT_PROGRAM_M , 'sweep_data'))

        self.create_output_directories()

    def init_output_path(self,scale_data,K,program,comment):

        root=op.join(self.DIR_OUT_PROGRAM,self.DIR_OUT_PROGRAM_M)
        self.DIR_SIM=op.join(root,'N_'+str(K.G.number_of_nodes())+'E_'+str(K.G.number_of_edges())+'_'+str(scale_data[0])+'x'+str(scale_data[1])+'_mode_'+K.graph_mode+comment)
        self.BACKUP=self.DIR_SIM
        self.create_output_paths(program)

    def init_kirchhoff_data(self,scale_data,parameters,K):

        OUTPUT_C=K.C[:]
        OUTPUT_S=K.J[:]
        K.E=np.zeros(int(scale_data[0]/scale_data[2]))

        return OUTPUT_C,OUTPUT_S

    def save_kirchhoff_data(self,O1,O2,K):

        B,BT=K.get_incidence_matrices()
        pos=nx.get_node_attributes(K.G,'pos')
        XYZ=np.array([pos[v] for v in K.G])
        export=[['C_temp',O1],['S_temp',O2],['E_temp',K.E],['F_final',K.F],['V_final',K.V],['IncidenceMatrix',B],['Pos',XYZ]]
        # export=[['C_temp',O1],['F_final',K.F],['V_final',K.V],['IncidenceMatrix',B],['Pos',XYZ]]
        for p in export:
            np.save(op.join(self.DIR_OUT_DATA,p[0]),p[1])

    def save_bilayer_kirchhoff_data(self,O1,O2,K):

        for i in range(len(K)):
            B,BT=K[i].get_incidence_matrices()
            pos=nx.get_node_attributes(K[i].G,'pos')
            XYZ=np.array([pos[v] for v in K[i].G])
            # export=[['C_temp',O1[i]],['S_temp',O2[i]],['E_temp',K[i].E],['F_final',K[i].F],['V_final',K[i].V],['IncidenceMatrix',B],['Pos',XYZ]]
            export=[['C_temp',O1[i]],['F_final',K[i].F],['V_final',K[i].V],['IncidenceMatrix',B],['Pos',XYZ]]
            for p in export:
                np.save(op.join(self.DIR_OUT_DATA,p[0]+'_'+str(i)),p[1])

    def load_kirchhoff_setting(self,set_file):

        input_file=open(op.join(self.DIR_SETTING,set_file))
        setting_dict = {}
        for line in input_file:
            info = line.split('=')
            if info[1].rstrip() == 'True':
                setting_dict.update ( {info[0]: True} )
            else:
                setting_dict.update ( {info[0]: False} )
        return setting_dict

    def save_runtime_parameters(self,parameters,scale_data):

        np.save(op.join(self.DIR_OUT_DATA,'parameters'),parameters)
        np.save(op.join(self.DIR_OUT_DATA,'scale_data'),scale_data)

class output_organizer_kirchhoff_cluster(output_organizer_kirchhoff,object):

    def __init__(self):
        super(output_organizer_kirchhoff_cluster,self).__init__()
        self.DIR_OUT = '/projects/project-kramer/'
        self.MEASURE_SETTING=[]

    def init_output_directory(self,K,program,jobid):

        self.DIR_OUT_PROGRAM=op.join(self.DIR_OUT , program)
        self.DIR_OUT_PROGRAM_M=op.join(self.DIR_OUT_PROGRAM_M,jobid)
        self.DIR_OUT_PROGRAM_PLOT=op.join( self.DIR_OUT_PROGRAM , op.join(self.DIR_OUT_PROGRAM_M , 'sweep_eval'))
        self.DIR_OUT_PROGRAM_DATA=op.join( self.DIR_OUT_PROGRAM , op.join(self.DIR_OUT_PROGRAM_M , 'sweep_data'))

        self.create_output_directories()

class output_organizer_liver_cluster(output_organizer,object):

    def __init__(self):
        super(output_organizer_liver_cluster,self).__init__()
        self.DIR_OUT = '/projects/project-kramer/'
        self.MEASURE_SETTING=[]

    def init_output_directory(self,program,jobid):

        self.DIR_OUT_PROGRAM=op.join(self.DIR_OUT , program)
        self.DIR_OUT_PROGRAM_M=op.join(self.DIR_OUT_PROGRAM_M,jobid)
        self.DIR_SIM=op.join(self.DIR_OUT_PROGRAM,self.DIR_OUT_PROGRAM_M)
        self.create_dir(self.DIR_SIM)

    def init_output_subdirectory(self,parameter):

        self.DIR_SIM=op.join(self.DIR_SIM,parameter)
        self.create_dir(self.DIR_SIM)

    def init_code_backup(self,loc_path,program):

        dst = os.path.join( self.DIR_SIM , 'source_code' )
        # self.create_dir(dst)
        if not op.exists(dst):
            try:
                wildcard_expression = op.join(loc_path,"*py")
                src = glob.glob(wildcard_expression)
                src.append(op.join(os.getcwd(),program))
                os.makedirs(dst)
                for f in src:
                    shutil.copy(f,dst)

            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
