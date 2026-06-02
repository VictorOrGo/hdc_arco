CC = gcc

CFLAGS = -Wall -O2 -Iinclude/

LDFLAGS = -lm -lcjson

DEFS =

SRC = src/hdc.c src/bytehd_lib.c src/bytehd_pc.c
OBJ = $(SRC:.c=.o)

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS)
	chmod +x $@


%.o: %.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
