import dca
import Bio
import os
import numpy as np
import sequenceHandler as sh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from math import ceil

''' 
This file contains the function plotTopContacts that plots the top N ranked DCA predictions overlaip on the structual contact map
originally written by the Duccio Malinverni https://doi.org/10.1007/978-1-4939-9608-7_16.
several modifications have been made to the original code.
'''

#the main function is at the end of the file and is called plotTopContacts
#this function uses apply_coordinates_squares that applies the coordinatesSquare to the error map dm_error and returns the new error map with null values outside the square


def apply_coordinates_squares(dm_error,coordinatesSquare):
    ''' 
    This function applies the coordinatesSquare to the error map dm_error and returns the new error map with null values outside the square
    input:
            dm_error           ->      error map
                                        np.array
            coordinatesSquare  ->      coordinates of the square area where we want to apply the error map
                                        list of 4 elements [x0,y0,weight,height]
                                        x0 and y0 are the coordinates of the top left corner of the square
                                        weight is the width of the square
                                        height is the height of the square
    output:
            dm_error_new       ->      new error map with null values outside the square
    '''
    if len(coordinatesSquare)!=4:
        print("coordinatesSquare should be a list of 4 elements")
        return
    [x_s,y_s,w_s,h_s]=coordinatesSquare
    min=np.min(dm_error[x_s-1:x_s-1+w_s+1,y_s-1:y_s-1+h_s+1])
    dm_error_new=np.ones(dm_error.shape)*min
    dm_error_new[(x_s-1):(x_s-1+w_s+1),y_s-1:(y_s-1+h_s+1)]=dm_error[(x_s-1):(x_s-1+w_s+1),(y_s-1):(y_s-1+h_s+1)]
    return dm_error_new


###########################################################################################################################################
#********************************************** MAIN FUNCTION *****************************************************************************
########################################################################################################################################### 

def plotTopContacts(pdbMap,dcaFile,Ntop,contactThreshold,pdbMap2='None', pdbDimer='None',pdbError='None',OnlyError=False,cutError_L=0,penalizingErrors=False,Nsquare=4,sigma=1,WithoutGauss=False,length_prot1=0,output_name="/",minSeqSeparation=4,OnlySquare=True,coordinatesSquare1=[0,0,0,0],coordinatesSquare2=[0,0,0,0],coordinatesSquare3=[0,0,0,0],coordinatesSquare4=[0,0,0,0],coordinatesSquare5=[0,0,0,0],coordinatesSquare6=[0,0,0,0],coordinatesSquare7=[0,0,0,0]):
    """ Plots the top N ranked DCA predictions overlaip on the structual contact map
    This functions plots the top N_top ranked DCA predictions overlaip on the structual contact map given by pdbMap.
    
    Input:

            - pdbMap            ->      path where to find the file containing the structual contact map
                                        if you want to visualise only the pdbDimer, please write None for this parameter
                                        string
                                        You can specify "None" if you want to plot only the Dimer 
            - dcaFile           ->      path where to find the file containing the DCA predictions
                                        string
            - Ntop              ->      number of top DCA predictions to plot
                                        int
            - contactThreshold  ->      distance threshold between two amino acids that is still considered as a contact
                                        float

    Default inputs:

            - pdbMap2           ->      a second pdbMap to compare the contacts of the two pdbMaps
                                        (for example if we have a dimer (A,B) and a protein (C) we could want to have a superposition of the maps AC and BC
                                        string, default='None'
            - pdbDimer          ->      a pdbMap of the dimer part -> in this case we will see the half of the plot with the dimer and the other hald with the monomer
                                        string, default='None'
            - pdbError          ->      path where to find the file containing the error map
                                        string, default='None'
            - OnlyError         ->      if True we plot only the error map
                                        boolean, default=False
            - cutError_L        ->      if cutError_L!=0 and penalizingErrors=True, we will not penalise the diagonal (+-cutError_L) of the error map 
                                        the errors in the diagonal (+- cutError_L) will be set to the minimum of the error map
                                        int, default=0
            - penalizingErrors  ->      if True we penalize the contacts with the error map
                                        boolean, default=False
            - Nsquare           ->      Number of square neighbors to consider for the gaussian filter during the penalization
                                        int, default=4  
            - sigma             ->      sigma of the gaussian filter
                                        float, default=1
            - WithoutGauss      ->      if True we don't use the gaussian filter for the penalization
                                        boolean, default=False
            - length_prot1      ->      if we have two proteins together and the couplings that come from a cross linear model
                                        we should specify the length of the first protein
                                        int, default=0
            - output_name       ->      path where to save the plot
                                        string, default= path of the dcaFile + '/<Ntop>topContacts-<contactThreshold>contactThreshold' + others parameters that are not default + '.png'
            - minSeqSeparation  ->      how much of the diagonal we will not consider
                                        int, default=4  
            - coordinatesSquare1->      coordinates of the first square area where we want to apply the error map and the penalization
                                        list of 4 elements [x0,y0,weight,height], default=[0,0,0,0]
                                        x0 and y0 are the coordinates of the top left corner of the square
                                        weight is the width of the square
                                        height is the height of the square

            - it's possible to give other coordinatesSquare  (max 6) with the same format as coordinatesSquare1

    Output:
            - plot of the structual contact map with the top Ntop DCA predictions overlaid with path output_name
    """
    
    print("---------------------------------------------------")
    print("--------- Welcome in PlotTopContacts :) ------------")
    print("Parameters:")
    print(f"contactThreshold={contactThreshold}")
    print(f"Ntop contacts to extract={Ntop}")
    print(f"Diagonal elements not considered={minSeqSeparation}")
    if length_prot1!=0:
        print(f"case of model linear cross proteins with length_prot1={length_prot1}")
    if penalizingErrors:
        if WithoutGauss:
            print("Penalizing errors without a gaussian filter")
        else:
            print(f"Penalizing errors with a gaussian filter of sigma={sigma} and Nsquare={Nsquare}")
    print("---------------------------------------------------")
    print("---------------------------------------------------")

    ##################################################################################
    ###################### load the pdb map and the dca file #########################
    ##################################################################################

    ######################### for a pdbDimer case ######################################
    if pdbDimer!='None':
        if pdbMap2!='None':
            print("Oups! You can't use pdbMap2 and pdbDimer at the same time...")
            print("pdbMap2 is when you have two disctinct protein with one that is a dimer.")
            print("pdbDimer is when you have only one type of protein that is a dimer.")
            print("Please try again...")
            return
        print("---------------- Monomer and dimer -----------------")
        dm_dimer=np.loadtxt(pdbDimer,dtype=float)
        shape_dimer=dm_dimer.shape[0]
        shape_monomer=dm_dimer.shape[0]//2
        len_y=shape_monomer-1
        len_x=shape_monomer-1
        print("shape_monomer=",shape_monomer)
        print("shape_dimer=",shape_dimer)

        dm_dimer_part1=dm_dimer[:shape_monomer,:shape_monomer] #only the monomer
        dm_dimer_part2=dm_dimer[:shape_monomer,shape_monomer:] #the intersection between the two monomers
        
        if pdbMap!='None':
            dm_monomer=np.loadtxt(pdbMap,dtype=float)
            dm_dimer=np.minimum(dm_monomer,dm_dimer_part2)
            #put in triangular form
            dm_monomer=np.triu(dm_monomer) #triangular sup
            dm_dimer=np.tril(dm_dimer) #triangular inf
            dm_monomer[dm_monomer==0]=np.inf 
            dm_dimer[dm_dimer==0]=np.inf 
            pdbContacts=np.argwhere(dm_monomer<=contactThreshold) #keep only the ones with a distance smaller than contactThreshold
            pdbContacts_dimer=np.argwhere(dm_dimer<=contactThreshold)
            pdbContacts_dimer[:,0],pdbContacts_dimer[:,1]=pdbContacts_dimer[:,1],pdbContacts_dimer[:,0].copy()
            pdbContacts[:,0],pdbContacts[:,1]=pdbContacts[:,1],pdbContacts[:,0].copy()
            print(f"Number of contacts by pdbMap with a distance smaller than {contactThreshold}: {len(pdbContacts)}")
            print(f"Number of contacts by pdbDimer with a distance smaller than {contactThreshold}: {len(pdbContacts_dimer)}")  
        else:
            print("No pdbMAP given so we will only plot the dimer")
            dm=np.minimum(dm_dimer_part1,dm_dimer_part2)
            pdbContacts=np.argwhere(dm<=contactThreshold)
            pdbDimer='None'
            print(f"Number of contacts by pdbDimer with a distance smaller than {contactThreshold}: {len(pdbContacts)/2}")
    else:
        dm=np.loadtxt(pdbMap,dtype=float)
        if length_prot1!=0 and pdbError!='None':
            dm[0:length_prot1,0:length_prot1]=np.inf
            dm[length_prot1:,length_prot1:]=np.inf
        pdbContacts=np.argwhere(dm<=contactThreshold)
        print(f"Number of contacts by pdbMap with a distance smaller than {contactThreshold}: {len(pdbContacts)/2}")
        len_y=dm.shape[1]-1
        len_x=dm.shape[0]-1
    ##################################################################################
    if pdbMap2!='None':  
        print("--------------------- Two Maps ---------------------")
        print("***USE FOR COUPLINGS EXTRACTED FROM A CROSS LINEAR MODEL***")
        dm2=np.loadtxt(pdbMap2,dtype=float)
        if length_prot1!=0 and pdbError!='None':
            dm2[0:length_prot1,0:length_prot1]=np.inf
            dm2[length_prot1:,length_prot1:]=np.inf
        pdbContacts2=np.argwhere(dm2<=contactThreshold)
    ##################################################################################
    ##################################################################################
    ##################################################################################
    

    ############################################################################################################
    ###################### load the error map and apply the coordinatesSquare if needed ########################
    ############################################################################################################
    # Overlay DCA predictions
    if pdbError!='None':
        print("---------------- Errors map given ------------------")
        dm_error = np.loadtxt(pdbError, dtype=float)
        #We will stock a list of errors maps to be prepared if we have several coordinatesSquares to apply
        coordinatesSquares=[]
        dm_errors=[]
        for i in range(6):
            coordinatesSquare=locals()["coordinatesSquare"+str(i+1)]
            [xs,ys,ws,hs,Ns]=coordinatesSquare
            xs=int(xs)
            ys=int(ys)
            ws=int(ws)
            hs=int(hs)
            Ns=int(Ns)
            if xs!=0 or ys!=0 or ws!=0 or hs!=0:
                print(f"coordinatesSquare{i+1}=({xs},{ys},{ws},{hs})")
                coordinatesSquares.append([xs,ys,ws,hs,Ns])
                #call the function apply_coordinates_squares that will return the error map with null values outside the square
                dm_error_new=apply_coordinates_squares(dm_error,[xs,ys,ws,hs])
            else: #we don't have any coordinatesSquare to apply
                if i>0:
                    dm_error_new=np.array([None]*len(dm_error)) #array of same shape but with only None
                else:
                    print("We will look all the errors of the map...")
                    print("If you want to try again but with only a certain area of the map, please give the coordinatesSquare1=[x0,y0,weight,height] parameter.")
                    dm_error_new=dm_error  #We have no coordinatesSquares to apply     
            dm_errors.append(dm_error_new)

        normalized_dm_errors=[] #list of normalized error maps
        for dm_error in dm_errors:
            if dm_error.all()==None: #for coordinatesSquare that are not used
                normalized_dm_errors.append(np.array([None]*len(dm_error)))
                continue
            if length_prot1==0:
                    min_err=np.min(dm_error)
                    max_err=np.max(dm_error)
                    normalized_dm_error= (dm_error - min_err) / (max_err - min_err)
                    normalized_dm_errors.append(normalized_dm_error)  
            else:
                min_err=np.min(dm_error[length_prot1:,:length_prot1])
                for i in range(len(dm_error)):
                    for j in range(len(dm_error)):
                        if i<length_prot1 and j<length_prot1:
                            dm_error[i,j]=min_err
                        if i>=length_prot1 and j>=length_prot1:
                            dm_error[i,j]=min_err
                max_err=np.max(dm_error)
                normalized_dm_error= (dm_error - min_err) / (max_err - min_err)
                normalized_dm_errors.append(normalized_dm_error)
    else:
        print("--------------- No Errors map given ----------------")
        normalized_dm_errors=None
        coordinatesSquares=[]
    ############################################################################################################
    ############################################################################################################


    #####################################################################
    ################## about the prediction of the DCA ##################
    #####################################################################
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("Extracting the top contacts from the DCA predictions...")
    dcaContacts,_,errors_maps,Ntop=dca.extractTopContacts(dcaFile,Ntop,minSeqSeparation,normalized_dm_errors,penalizingErrors,Nsquare,sigma,WithoutGauss,coordinatesSquares,cutError_L,length_prot1)
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("Extracting the good and wrong contacts from the DCA predictions...")
    if pdbMap2!='None':
        dcaColors1=dm[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold
        dcaColors2=dm2[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold
        dcaColors=['lime' if (col1 or col2) else 'red' for col1,col2 in zip(dcaColors1,dcaColors2)]
    else:
        if pdbDimer=='None':
            dcaColors=dm[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold 
            dcaColors=['lime' if col else 'red' for col in dcaColors]
        else:
            dcaColors_dimer=dm_dimer[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold
            dcaColors_monomer=dm_monomer[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold
            dcaColors=['lime' if (col1 or col2) else 'red' for col1,col2 in zip(dcaColors_dimer,dcaColors_monomer)]
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    #####################################################################
    #####################################################################
    #####################################################################


    #########################################################################################################
    ################################## Plot the contacts of the DCA #########################################
    #########################################################################################################
    figure_size_x=6
    figure_size_y=7
    fig, ax = plt.subplots(figsize=(figure_size_x,figure_size_y))
    #### properties of the plot to adapt the size of the scatter points as the size of the figure changes ##
    height_axis_figure_old=len_y
    width_axis_figure_old=len_x
    x_step_size=1.0/len_x
    y_step_size=1.0/len_y
    rect_size_x_old=x_step_size*figure_size_x*100/2
    rect_size_y_old=y_step_size*figure_size_y*100/2
    #########################################################################################################

    ########################  addaptive size of the scatter points #############################################
    # Connect to the axes 'on_changed' event to dynamically update the scatter plot
    def on_change(event):
        minY_new, maxY_new = ax.get_ylim() #this is not correct
        minX_new, maxX_new = ax.get_xlim()
        height_axis_figure_new=-maxY_new+minY_new
        width_axis_figure_new=maxX_new-minX_new
        #find the new size of the square
        proportion_zoom_y=height_axis_figure_new/height_axis_figure_old
        proportion_zoom_x=width_axis_figure_new/width_axis_figure_old
        square_size=(rect_size_x_old/proportion_zoom_x)*(rect_size_y_old/proportion_zoom_y)
        scatter_contact_pred.set_sizes([square_size]*len(dcaContacts))
        scatter_contact_true1.set_sizes([square_size]*len(pdbContacts))
        if pdbMap2!='None':
            scatter_contact_true2.set_sizes([square_size]*len(pdbContacts2))
        if pdbDimer!='None':
            scatter_Dimmer.set_sizes([square_size]*len(pdbContacts_dimer))
        plt.draw()
    ############################################################################################################

    if not OnlyError:
        ###################### PLOTS THE MAPS AND PREDICTIONS #######################################
        #Scatter the contacts of the pdbmap with the good size (in black)
        #We add 1 to the coordinates because the scatter plot starts at 1 and not at 0
        if pdbDimer!='None':
            dcaContacts[:,0],dcaContacts[:,1]=dcaContacts[:,1],dcaContacts[:,0].copy()
        scatter_contact_true1=ax.scatter(pdbContacts[:,0]+1,pdbContacts[:,1]+1,s=rect_size_x_old*rect_size_y_old,color='black',alpha=0.4,marker='s')
        if pdbMap2!='None': #we should superpose the second map2
            scatter_contact_true2=ax.scatter(pdbContacts2[:,0]+1,pdbContacts2[:,1]+1, s=rect_size_x_old*rect_size_y_old,color='gray',alpha=0.4,marker='s')
        #Plot the predictions
        scatter_contact_pred = ax.scatter(dcaContacts[:,0]+1, dcaContacts[:,1]+1, color=dcaColors, s=rect_size_x_old*rect_size_y_old, marker='s')
        ################################################################################################
        ## print the numbers of wrong and correct contacts ##
        if pdbDimer=='None':
            ####### print of the text on the plot with the percentage ######
            if pdbError!='None':
                dx=20
            else:
                dx=5
            print("Number of wrong contacts: "+str(dcaColors.count('red')))
            print("Number of correct contacts: "+str(dcaColors.count('lime')))
            print(Ntop)
            
            if not OnlySquare:
                if length_prot1!=0:
                    x0_text_1=((len_x-length_prot1+1)//2-(len_x-length_prot1+1)//4)+(length_prot1)
                    x0_text_2=((len_x-length_prot1+1)//2+(len_x-length_prot1+1)//4)+(length_prot1)
                else:
                    x0_text_1=(len_x+1)//2 -(len_x+1)//4
                    x0_text_2=(len_x+1)//2 +(len_x+1)//4
                y0_text=0
                if pdbMap2!='None':
                    #we have a big map
                    y0_text-=15
            if OnlySquare: #we have only one coordinatesSquare1 and we want only visualise it and not the whole map
                x0_text_1=int(coordinatesSquare1[2])//2-int(coordinatesSquare1[2])//4 + coordinatesSquare1[0]
                x0_text_2=int(coordinatesSquare1[2])//2+int(coordinatesSquare1[2])//4 + coordinatesSquare1[0]
                y0_text=coordinatesSquare1[1]
            ax.text(x0_text_1,y0_text -22, f'Wrong {(dcaColors.count("red"))/(Ntop)*100:.0f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='red', alpha=0.5))
            ax.text(x0_text_2,y0_text-22, f'Correct {(dcaColors.count("lime"))/(Ntop)*100:.0f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='lime', alpha=0.5))
            ################################################################
            maxX=dm.shape[0]
            maxY=dm.shape[1]

        ################################################################################################
        ############################## DIMER CASE ######################################################
        else: #we have a dimer -> need a plot divided into two parts (with the dimer and the monomer)
            #color the half diagonal plot in black so only for x<=y
            dots_to_fill=np.zeros_like(dm_monomer)
            for j in range(len(dm_monomer)):
                for i in range(j,len(dm_monomer),1):
                    dots_to_fill[i,j]=1
            #fill with black
            ax.imshow(dots_to_fill, cmap='binary')
            scatter_Dimmer=ax.scatter(pdbContacts_dimer[:,0]+1,pdbContacts_dimer[:,1]+1,s=rect_size_x_old*rect_size_y_old,color='gray',marker='s')
            scatter_contact_pred = ax.scatter(dcaContacts[:,0]+1, dcaContacts[:,1]+1, color=dcaColors, s=rect_size_x_old*rect_size_y_old, marker='s') 
            ax.text((len_x+1)//4, len_y//2, 'Dimer', color='white', ha='center', rotation=90, fontsize=15)
            ax.text((len_x+1)//2, len_y//4, 'Monomer',  color='black', ha='center', fontsize=15)
            plt.plot([1,shape_monomer+1],[1,shape_monomer+1],color='k')
            nb_wrong_in_monomer,nb_prediction_in_dimer,nb_wrong_in_dimer,nb_prediction_in_monomer=0,0,0,0 #initialization
            for contact in dcaContacts:
                #look if it in triangular sup of the matrix
                if contact[1]>=contact[0]: 
                    nb_prediction_in_monomer+=1
                    if dm_monomer[contact[0],contact[1]]>contactThreshold:
                        nb_wrong_in_monomer+=1
                        
                #look if it in triangular inf of the matrix
                if contact[1]<=contact[0]:
                    nb_prediction_in_dimer+=1
                    if dm_dimer[contact[0],contact[1]]>contactThreshold:
                        nb_wrong_in_dimer+=1        
            P_w_monomer=nb_wrong_in_monomer/nb_prediction_in_monomer*100
            P_w_dimer=nb_wrong_in_dimer/nb_prediction_in_dimer*100
            P_w_total=(nb_wrong_in_monomer+nb_wrong_in_dimer)/(nb_prediction_in_monomer+nb_prediction_in_dimer)*100
            print(f"Number of wrong contacts in monomer: {str(nb_wrong_in_monomer)}/{str(nb_prediction_in_monomer)} (percentage:{str(P_w_monomer)}) ")
            print(f"Number of wrong contacts in dimer: {str(nb_wrong_in_dimer)}/{str(nb_prediction_in_dimer)} (percentage:{str(P_w_dimer)}) ")
            print(f"Number total of wrong contact: {str(nb_wrong_in_monomer+nb_wrong_in_dimer)}/{str(nb_prediction_in_monomer+nb_prediction_in_dimer)} (percentage:{str(P_w_total)}) ")
            ####### print of the text on the plot with the percentage ######
            ax.text((len_x+1)//2-(len_x+1)//4, -14.5, f'Wrong {P_w_monomer:.1f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='red', alpha=0.5))
            ax.text((len_x+1)//2+(len_x+1)//4,-14.5, f'Correct {(100-P_w_monomer):.1f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='lime', alpha=0.5))
            ax.text((len_x+1)//2-(len_x+1)//4, len_y+62.5, f'Wrong {P_w_dimer:.1f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='red', alpha=0.5))
            ax.text((len_x+1)//2+(len_x+1)//4,len_y+62.5, f'Correct {100-P_w_dimer:.1f}%', fontsize=12, color='black', ha='center', bbox=dict(facecolor='lime', alpha=0.5))
            ################################################################
            maxX=shape_monomer
            maxY=shape_monomer
        ################################################################################################
        
        ax.callbacks.connect('xlim_changed', on_change)
        ax.callbacks.connect('ylim_changed', on_change)
    else:
        maxX=dm.shape[0]
        maxY=dm.shape[1]
    ################### add the error map ############################################################
    if pdbError!='None':
        #colors= ["white","pink","orchid","moccasin","gold","orange","palegreen","yellowgreen","green","darkcyan","dodgerblue","cyan","steelblue","mediumblue"]
        #colors= ["white","silver","gray","tan","gold","goldenrod","steelblue","teal","mediumblue","black"]
        colors=['white','yellow','orange','paleturquoise','paleturquoise','lightskyblue','gray','purple','black']
        cmap_name = 'custom_cbar'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors)
        # Plot the heatmap with the custom colormap and normalized data
        L=len_y+1
        if coordinatesSquares!=[]: #we have several normalized_dm_error in normalized_dm_errors
            errors_maps_total=errors_maps[0]
            for id,normalized_dm_error in enumerate(normalized_dm_errors):
                if normalized_dm_error.all()==None:
                    continue
                x_s,y_s,w_s,h_s,_=coordinatesSquares[id]
                x_min=int(x_s-1)
                y_min=int(y_s-1)
                x_max=int(x_s+w_s-1)
                y_max=int(y_s+h_s-1) 
                x_s=int(x_s)
                y_s=int(y_s)
                if penalizingErrors and not WithoutGauss: #we don't plot normalized_dm_error but errors_maps
                    colors=['white','yellow','orange','paleturquoise','paleturquoise','lightskyblue','gray','purple','black']
                    cmap_name = 'custom_cbar'
                    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
                    #add a square that has the corner left at(x_s-1+ceil(Nsquare/2),y_s-1+ceil(Nsquare/2)) and the corner right at (x_s-1+w_s+1-ceil(Nsquare/2),y_s-1+h_s+1-ceil(Nsquare/2))
                    rect1=plt.Rectangle((x_s+ceil(Nsquare/2)-0.5, y_s+ceil(Nsquare/2)-0.5),w_s-2*ceil(Nsquare/2)+1, h_s-2*ceil(Nsquare/2)+1, linewidth=1, edgecolor='white', facecolor='none')
                    plt.gca().add_patch(rect1)

                #problem with the extent....
                errors_map=errors_maps[id]
                #we want to have values between 0 and 1 for error_map[x_min:x_max+1,y_min:y_max+1]
                errors_map[x_min:x_max+1,y_min:y_max+1]=(errors_map[x_min:x_max+1,y_min:y_max+1]-np.min(errors_map[x_min:x_max+1,y_min:y_max+1]))/(np.max(errors_map[x_min:x_max+1,y_min:y_max+1])-np.min(errors_map[x_min:x_max+1,y_min:y_max+1]))
                #complete the errors_maps with values if elements are not zero
                for i in range(x_min,x_max+1):
                    for j in range(y_min,y_max+1):
                        if errors_maps_total[i,j]==0 and errors_map[i,j]!=0:
                            errors_maps_total[i,j]=errors_map[i,j]
            extend=[1-0.5,L-0.5+1,L-0.5+1,1-0.5]#left, right, bottom,top
            plt.imshow(errors_maps_total.T, cmap=cm, extent=extend)
            max_color=np.max(errors_maps_total)          
        else: #we will only have one normalized_dm_error in normalized_dm_errors and apply it to the whole map
            extend=[1-0.5,L-0.5+1,L-0.5+1,1-0.5]#left, right, bottom,top
            print("L IS :",L)
            if penalizingErrors: #we don't plot normalized_dm_error but errors_maps
                errors_map=errors_maps[0]
                errors_map-=np.min(errors_map)
                if not WithoutGauss:
                    #colors=['white','blue','cyan','magenta','yellow','orange','pink','gray','black']
                    colors=['white','gold','yellow','orange','pink','chocolate','sienna','brown','green','maroon','purple','darkviolet','black']
                    cmap_name = 'custom_cbar'
                    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
                    rect1=plt.Rectangle((0+ceil(Nsquare/2)+0.5, 0+ceil(Nsquare/2)+0.5), L-2*ceil(Nsquare/2), L-2*ceil(Nsquare/2), linewidth=1, edgecolor='white', facecolor='none')
                    plt.gca().add_patch(rect1)
                plt.imshow(errors_map, cmap=cm, extent=extend)
                max_color=np.max(errors_map)
            else:
                normalized_dm_error=normalized_dm_errors[0]
                normalized_dm_error-=np.min(normalized_dm_error)
                plt.imshow(normalized_dm_error, cmap=cm, extent=extend)  
                max_color=np.max(normalized_dm_error)        
        
        #### add the line of the cutError_L for the border ###
        if cutError_L!=0:
            #plot the line of the cutError_L for the border
            x_0_line=cutError_L
            x_1_line=L
            y_0_line=0
            y_1_line=L-cutError_L
            #plot a line going from (x_0_line,y_0_line) to (x_1_line,y_0_line)
            plt.plot([x_0_line,x_1_line],[y_0_line,y_1_line],color='darkgreen',linewidth=2)
            plt.plot([y_0_line,y_1_line],[x_0_line,x_1_line],color='darkgreen',linewidth=2)
        #########################################################
    
        #add the color bar
        if not OnlySquare:
            if pdbMap2!='None':
                if length_prot1!=0:
                    cax = ax.inset_axes([length_prot1-0.5,0-50.5, L-length_prot1, 10], transform=ax.transData)
                else:
                    cax = ax.inset_axes([coordinatesSquare1[0]-0.5, -20,coordinatesSquare1[2],5], transform=ax.transData)
            else:
                #cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-40.5,coordinatesSquare1[2],5], transform=ax.transData) #position of the color bar with [x0,y0,w,h
                cax = ax.inset_axes([1-0.5, 0-100.15,L,15], transform=ax.transData) #position of the color bar with [x0,y0,w,h
        else:
            if figure_size_y==7:
                if figure_size_x==6:
                    cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-0.5-35.5, coordinatesSquare1[2], 25], transform=ax.transData)
                elif figure_size_x==7:
                    cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-0.5-25.5, coordinatesSquare1[2], 10], transform=ax.transData)
                elif figure_size_x==9:
                    cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-0.5-45.5, coordinatesSquare1[2], 10], transform=ax.transData)
                else:
                    cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-0.5-55.5, coordinatesSquare1[2], 20], transform=ax.transData)
            else:
                cax = ax.inset_axes([coordinatesSquare1[0]-0.5, coordinatesSquare1[1]-45.5, coordinatesSquare1[2], 10], transform=ax.transData)

        cbar = plt.colorbar(label='Percentage of maximum error', orientation='horizontal', cax=cax)

        cbar.set_ticks([0,0.2*max_color,0.4*max_color,0.6*max_color,0.8*max_color,max_color])
        cbar.set_ticklabels(['0%','20%','40%', '60%', '80%', '100%'])
        cbar.set_label('Percentage of maximum error', fontsize=15)
        
        # Move the ticks and label to the top
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=12) 
        


    #temporary
    rect2=plt.Rectangle((1-0.5, 455-0.5), 452+1, 178+1, linewidth=2, edgecolor='turquoise', facecolor='none')
    #plt.gca().add_patch(rect2)
    rect3=plt.Rectangle((455-0.5, 455-0.5), 178+1, 178+1,  linewidth=2, edgecolor='blue', facecolor='none')
    #plt.gca().add_patch(rect3)
    rect1=plt.Rectangle((1-0.5, 1-0.5), 450+1, 450+1, linewidth=2, edgecolor='purple', facecolor='none')
    #plt.gca().add_patch(rect1)
    rect5=plt.Rectangle((1-0.5, 670-0.5), 448+1, 329+1,linewidth=2,edgecolor='lime',facecolor='none')
    #plt.gca().add_patch(rect5)
    rect2=plt.Rectangle((1-0.5, 670-0.5), 448+1, 329+1, linewidth=2, edgecolor='red', facecolor='none')
    #plt.gca().add_patch(rect2)
    rect2=plt.Rectangle((670-0.5, 670-0.5), 130+1, 130+1, linewidth=2, edgecolor='pink', facecolor='none')
    #plt.gca().add_patch(rect2)
    rect2=plt.Rectangle((801-0.5, 801-0.5), 198+1, 198+1, linewidth=2, edgecolor='yellow', facecolor='none')
    #plt.gca().add_patch(rect2)

    if length_prot1!=0:
        plt.ylim([0.5,length_prot1+0.5])
        plt.xlim([length_prot1+0.5,maxX+0.5])
    else:
        if not OnlySquare:
            plt.xlim([0.5,maxX+0.5])
            plt.ylim([0.5,maxY+0.5])
        else:
            plt.xlim([coordinatesSquare1[0]-0.5,coordinatesSquare1[0]+coordinatesSquare1[2]+0.5])
            plt.ylim([coordinatesSquare1[1]-0.5,coordinatesSquare1[1]+coordinatesSquare1[3]+0.5])
    plt.xlabel('Residue i',fontsize=15)
    plt.ylabel('Residue j',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.gca().invert_yaxis()
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################



    #######################################################################
    ################## Save the plot ######################################
    #######################################################################
    if output_name=="/":
        #take the same path than the dcaFile and add <Ntop>_<contactThreshold>.png
        output_name = os.path.join(os.path.dirname(dcaFile), f'{str(Ntop)}Pred_{str(contactThreshold)}Threshold.png')
    if penalizingErrors:
        if WithoutGauss:
            output_name = os.path.join(output_name.split(".png")[0]+f'_WithoutGauss.png')
        else:
            output_name = os.path.join(output_name.split(".png")[0]+f'_{str(sigma)}sigma_{str(Nsquare)}Nsquare.png')
        if cutError_L!=0:
            output_name = os.path.join(output_name.split(".png")[0]+f'_{str(cutError_L)}noPenalizedDiag.png')
    
    if minSeqSeparation>0:
        output_name = os.path.join(output_name.split(".png")[0]+f'_{str(minSeqSeparation)}ik.png')

    output_directory = os.path.dirname(output_name) # Path for the couplings directory without the last part of the output_name (to stock in the folder)
    os.makedirs(output_directory, exist_ok=True) # Create the directory if it does not exist
    #check if the file already exist
    try:
        with open(output_name, "r") as file: #"r" is for read
            #ask if we want to overwrite it
            overwrite=input("Do you want to overwrite it? (yes/no) ")
            while overwrite!="yes" and overwrite!="no":
                overwrite=input("Please enter yes or no")
            if overwrite=="no":
                R=input("Do you want to cancel the operation? If no you will write a new name for the file (yes/no)")
                while R!="yes" and R!="no":
                    input("Please enter yes or no")
                if R=="yes":
                    return
                else:
                    new_output_name = input("Please enter the new name of the output file (without the extension): ")
                    new_output_name = new_output_name + '.png'
    except:
        pass
    plt.savefig(output_name, dpi=300, bbox_inches='tight') # Save the figure in the output_name directory
    output_eps=output_name.split(".")[:-1]
    output_eps=".".join(output_eps)+".eps"
    #plt.savefig(output_eps, format='eps') # Save the figure in the output_name directory
    #######################################################################
    #######################################################################
    #######################################################################
    plt.show()
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

    