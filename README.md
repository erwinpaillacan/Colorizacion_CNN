
# Colorización de Imágenes
## Sin transfer learning
### Integrantes: Erwin Paillacán- Héctor Zúñiga

¿Cómo usar?  <br /> 
 
Dejar todas las imagenes que se quieran ocupar para entrenamiento y validacion en la carpeta: ```/train``` y ```/validation```. Las imágenes de prueba dejarlas en la Carpeta```/test```. Luego ejecutar sólo una vez ```data_utils_CNN.py```, que crea los tfrecord. Luego ejecutar ```train_CNN.py``` y finalmente ```evaluar_CNN.py``` que coloriza 100 imagenes de la carpeta test y las guarda en ```/results``` . Recomiendo ejecutar las cosas con Pycharm cerrado y Chrome igual, se liberan 3 Gb de RAM.<br />
 <br />
### Resultados
<img src="poster.png" width="750" height="650" /> <br />
