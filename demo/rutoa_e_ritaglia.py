import cv2
import numpy as np
import os
from mmocr.apis import MMOCRInferencer
import os

def rot_image(image, midpoint, angle_deg):

	"""# Convert the box corners to NumPy array for easier manipulation
	corners_array = np.array(box_corners, dtype=np.float32)

	# Calculate the midpoint of the bounding box
	midpoint = np.mean(corners_array, axis=0)"""


	#midpoint = (midpoint_x, midpoint_y)

	# Create an affine transformation matrix for rotation
	rotation_matrix = cv2.getRotationMatrix2D(tuple(midpoint), angle_deg, scale=1)

	# Apply the rotation to the image
	rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

	return rotated_image

def main(dir_jpg, dir_txt):
    dir_out = "./img_tagliate_ruotate1803/"
    immagini = [x for x in os.listdir(dir_jpg) if x.split(".")[1] == "jpg"]

    for img in immagini[0:10]:
        #img = '21RF110738-4638-430_B.jpg' 
        image = cv2.imread(dir_jpg + img)
        txtname = img.split(".")[0] + ".txt"
        
        with open(dir_txt+txtname,"r") as f:
            contenuto = f.read()
        
        coord = [float(numero) for numero in contenuto.split()]

        #NUMERATORE
        filename_num_out = dir_out + img.split(".")[0] + "_num_.jpg"
        center_x = coord[0]
        center_y = coord[1]
        width = coord[2]
        height = coord[3]
        angle_deg = np.degrees(coord[4])

        M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1)
        # Applica la trasformazione per ottenere i nuovi vertici della bounding box ruotata
        rotated_center = np.dot(M, np.array([center_x, center_y, 1]))

        rotated_image = rot_image(image, (center_x, center_y), angle_deg)
        cv2.imshow('Immagine ruotata: ', rotated_image)
        
        x_min = int(rotated_center[0] - width / 2)
        y_min = int(rotated_center[1] - height / 2)
        x_max = int(rotated_center[0] + width / 2)
        y_max = int(rotated_center[1] + height / 2)
        # Taglia l'immagine utilizzando gli angoli calcolati
        final_image = rotated_image[y_min-10:y_max+10, x_min-10:x_max+10]
        cv2.imshow('Immagine ruotata e tagliata: ', final_image)
        cv2.imwrite(filename_num_out, final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #DENOMINATORE
        filename_den_out = dir_out + img.split(".")[0] + "_den_.jpg"
        center_x = coord[6]
        center_y = coord[7]
        width = coord[8]
        height = coord[9]
        angle_deg = np.degrees(coord[10])
        M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1)
        # Applica la trasformazione per ottenere i nuovi vertici della bounding box ruotata
        rotated_center = np.dot(M, np.array([center_x, center_y, 1]))
        rotated_image = rot_image(image, (center_x, center_y), angle_deg)
        cv2.imshow('Immagine ruotata: ', rotated_image)
        x_min = int(rotated_center[0] - width / 2)
        y_min = int(rotated_center[1] - height / 2)
        x_max = int(rotated_center[0] + width / 2)
        y_max = int(rotated_center[1] + height / 2)
        # Taglia l'immagine utilizzando gli angoli calcolati
        final_image = rotated_image[y_min:y_max, x_min:x_max]
        cv2.imshow('Immagine ruotata e tagliata: ', final_image)
        cv2.imwrite(filename_den_out, final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == "__main__":
    # Carica l'immagine
    main("./res_inference_0703/","./res_inference_0703/")

