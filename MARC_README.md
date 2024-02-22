vanno create le seguenti cartelle e le immagini vanno messe dentro:
mmrotate/tools/data/marc/train/image
mmrotate/tools/data/marc/val/image

le immagini devono essere in formato png

inoltre vanno create le seguenti cartelle per le annotazioni 
mmrotate/tools/data/marc/train/annots
mmrotate/tools/data/marc/val/annots

e dentro inseriti file .txt con stesso nome dell'immagine.

con la seguente notazione:
278 487 415 577 373 640 236 551 num 0
319 341 524 475 475 550 270 415 den 0

i primi 8 punti sono coordinate dell'oggetto, poi nome classe e l'ultimo non ricordo.


link file config pth e file log da mettere nella cartella tools:
https://drive.google.com/file/d/1lFTuvRc3e8o_RyI3pi6CM0GDlbpC3i0T/view?usp=sharing
