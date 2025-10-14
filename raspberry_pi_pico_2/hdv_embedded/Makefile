# Makefile para tu proyecto

# Compilador
CC = gcc
CFLAGS = -Wall -O2 -Iinclude/
LDFLAGS = -lm -lcjson

# Archivos fuente y objetos
SRC = src/hdc.c src/bytehd.c src/hdv_embedded_pc.c
OBJ = $(SRC:.c=.o)

# Ejecutable final
TARGET = main

# Regla por defecto
all: $(TARGET)

# Compilación del ejecutable
$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS)
	chmod +x $@

# Regla para compilar objetos
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Limpiar archivos generados
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean

	