from mmocr.apis import MMOCRInferencer
import os
import cv2
import numpy as np

def marc_inferencer():
    models = ["SAR"]
    for mod in models:
        inferencer = MMOCRInferencer(rec=mod)
        for img in os.listdir('./img_tagliate_ruotate1803/'):
            inferencer(os.path.join('./img_tagliate_ruotate1803/', img), show=True, print_result=True, out_dir=f'./data/results/{mod}/cut_rot', save_pred=True, save_vis=True)


def cv2_recog():
    dir_in = "./img_tagliate_ruotate1803/"
    immagini = [x for x in os.listdir(dir_in) if x.split(".")[1] == "jpg"]
    print(immagini)
    pixels = 3
    for img in immagini:
        image = cv2.imread(dir_in+img)
        # Converti l'immagine in scala di grigi
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((pixels, pixels), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        # Applica il filtro di Canny per rilevare i contorni
        canny_image = cv2.Canny(dilated_image, 50, 150, apertureSize=3)

        # Trova i contorni nell'immagine rilevata da Canny
        contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Disegna i contorni trovati sull'immagine originale
#        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        # Seleziona il contorno con l'area maggiore
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

        # Disegna il contorno selezionato sull'immagine originale
            result_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)

        # Mostra l'immagine risultante
            cv2.imshow("Largest Contour", result_image)

        # Visualizza l'immagine con i contorni
        cv2.imshow('Objects Divided with Canny', image)
        cv2.imshow('Dilated Image', dilated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    marc_inferencer()
    #cv2_recog()

