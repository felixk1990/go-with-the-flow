# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   kramer
# @Last modified time: 23-06-2021

# general initial value problem for network morpgogenesis
class morph_ivp(  ):

    def __init__(self, flow):

        self.flow=flow

    #
    # def save_dynamic_output(self,*args):
    #
    #     items=args[0]
    #     keys=args[1]
    #     K=args[2]
    #
    #     dict_output={}
    #     for i,key in enumerate(keys):
    #         dict_output[key]=items[i]
    #
    #     K.set_network_attributes()
    #     nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
    #     f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
    #     pickle.dump(dict_output,f)
    #     f.close()
    #
    # # evaluate time series
    # def update_radii(self,R,K):
    #     K.R=R
    #     K.C=np.power(R,4)*K.k
    #
    # def evaluate_timeline(self,nsol,K,mode):
    #
    #     options=['absorption+shear','shear+absorption','absorption','shear','absorption+volumetric','absorption+volumetric+shear']
    #     func=[self.evaluate_absorption_shear, self.evaluate_shear_absorption,self.evaluate_absorption,self.evaluate_shear,self.evaluate_absorption_volumetric,self.evaluate_absorption_volumetric_shear]
    #     dict_output={}
    #     check=True
    #
    #     for i,op in enumerate(options):
    #         if op==mode:
    #             check=False
    #             dict_output=func[i](nsol,K)
    #             break
    #     if check:
    #         print(' nothing to do with my time')
    #
    #     return dict_output
    #
    # def evaluate_absorption_shear(self,nsol,K):
    #
    #     dict_output={}
    #     phi0=np.ones(self.M)*K.phi0
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #
    #         overall_uptake_diff=np.sum(np.power(np.subtract(phi,phi0),2))
    #         dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
    #         volume_penalty=np.sum( sq_R )
    #
    #         F.append( overall_uptake_diff+K.alpha_1*dissipation+K.alpha_0*volume_penalty )
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
    #
    # def evaluate_shear_absorption(self,nsol,K):
    #
    #     dict_output={}
    #     phi0=np.ones(self.M)*K.phi0
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #
    #         overall_uptake_diff=np.sum(np.power(np.subtract(phi,phi0),2))
    #         dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
    #         volume_penalty=np.sum( sq_R )
    #
    #         F.append( K.alpha_0* overall_uptake_diff+dissipation+K.alpha_1*volume_penalty )
    #
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
    #
    # def evaluate_absorption(self,nsol,K):
    #
    #     dict_output={}
    #     phi0=np.ones(self.M)*K.phi0
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #
    #         F.append(np.sum(np.power(np.subtract(phi,phi0),2)))
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
    #
    # def evaluate_absorption_volumetric(self,nsol,K):
    #
    #     dict_output={}
    #     ones=np.ones(len(K.dict_volumes.values()))
    #     phi0=ones*K.phi0
    #     dphi=ones
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #
    #         for j,v in enumerate(K.dict_volumes.keys()):
    #             dphi[j]=np.sum(phi[K.dict_volumes[v]])-phi0[j]
    #
    #         F.append(np.sum(np.power(dphi,2)))
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
    #
    # def evaluate_absorption_volumetric_shear(self,nsol,K):
    #
    #     dict_output={}
    #     ones=np.ones(len(K.dict_volumes.values()))
    #     phi0=ones*K.phi0
    #     dphi=ones
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #         dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
    #         volume_penalty=np.sum( sq_R )
    #
    #         for j,v in enumerate(K.dict_volumes.keys()):
    #             dphi[j]=np.sum(phi[K.dict_volumes[v]])-phi0[j]
    #
    #         F.append(np.sum(np.power(dphi,2))+ K.alpha_1*dissipation+K.alpha_0*volume_penalty)
    #
    #         # F.append( K.alpha_1*dissipation+K.alpha_0*volume_penalty )
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
    #
    # def evaluate_shear(self,nsol,K):
    #
    #     dict_output={}
    #     phi0=np.ones(self.M)*K.phi0
    #     F, PHI, C, PE =[],[],[],[]
    #
    #     for i in range(len(nsol[:,0])):
    #
    #         self.update_radii(nsol[i,:],K)
    #         sq_R=np.power(K.R,2)
    #
    #         c,B_new,K=self.calc_profile_concentration(K)
    #         phi=self.calc_absorption( K.R, K )
    #
    #         dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
    #         volume_penalty=np.sum( sq_R )
    #         F.append( K.alpha_1*dissipation+K.alpha_0*volume_penalty )
    #
    #         PHI.append(phi)
    #         C.append(c)
    #         PE.append(self.calc_PE(K))
    #
    #     dict_output['radii_temporal']=nsol
    #     dict_output['cost']=F
    #     dict_output['uptake']=PHI
    #     dict_output['concentration']=C
    #     dict_output['PE']=PE
    #
    #     return dict_output
