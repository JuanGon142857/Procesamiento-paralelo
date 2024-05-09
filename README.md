# Procesamiento-paralelo

utilidades.py tiene funciones varias que se utilizan en los otros códigos

proyecto_MPI4.py carga el holograma "multiplane_hologram.bmp" y genera secciones de un holograma binario optimizado que lo reproduce. El programa se basa en mpi4py y se corre en sus distintas formas usando "mpiexec -n N python proyecto_MPI4.py", donde N es un número al cuadrado. Las secciones se guardan en la carpeta Recuadros

pegar_recuadros.py toma las secciones del holograma generada con el código anterior, las junta en un holograma completo, los propaga a los respectivos planos y guarda las reconstrcciones. Las reconstrucciones se guardan en la carpeta Resultados

Articulo.pdf es el articulo con los resultados

La presentación de los resultados se encuentra en https://youtu.be/uoccB6PBTXk
