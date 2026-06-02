# ==============================
# Makefile para tu proyecto
# ==============================

# Compilador
CC = gcc

# Flags de compilación
CFLAGS = -Wall -O2 -Iinclude/

# Flags de enlace
LDFLAGS = -lm -lcjson

# Macros (se pueden pasar desde línea de comandos)
DEFS =

# Archivos fuente y objetos
SRC = src/hdc.c src/bytehd_lib.c src/hdclite_pc_v2.c
OBJ = $(SRC:.c=.o)

# Ejecutable final
TARGET = main

# ==============================
# Regla por defecto
# ==============================
all: $(TARGET)

# ==============================
# Link del ejecutable
# ==============================
$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS)
	chmod +x $@

# ==============================
# Compilación de objetos
# ==============================
%.o: %.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

# ==============================
# Limpieza
# ==============================
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean