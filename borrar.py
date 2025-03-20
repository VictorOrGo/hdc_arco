def borrar_desde_palabra(fichero, palabra_clave):
    try:
        # Leer todas las líneas del fichero
        with open(fichero, 'r') as archivo:
            lineas = archivo.readlines()
        
        # Buscar la palabra clave e identificar el índice
        indice = None
        for i, linea in enumerate(lineas):
            if palabra_clave in linea:
                indice = i
                break
        
        # Si se encontró la palabra clave, escribir solo las líneas antes de ella
        if indice is not None:
            with open("/home/victor/Descargas/new_file.json", 'w') as archivo:
                archivo.writelines(lineas[:indice])
            print(f"Se han eliminado las líneas desde la palabra clave '{palabra_clave}' inclusive.")
        else:
            print(f"La palabra clave '{palabra_clave}' no se encontró en el fichero.")
    
    except FileNotFoundError:
        print("El fichero no se encontró.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso
fichero = '/home/victor/Descargas/xmas_merged_data.json'  # Nombre del fichero
palabra_clave = 'STOP'  # Palabra clave para detenerse y borrar
borrar_desde_palabra(fichero, palabra_clave)