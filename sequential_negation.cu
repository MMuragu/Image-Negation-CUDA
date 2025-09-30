#include <stdio.h>
#include "image.h"

void process_data(image *photo) {
    int i, n;

    n = photo->width * photo->height * 3;

    for (i = 0; i < n; i++) {
        photo->data[i] = photo->max_value - photo->data[i];
    }
}

image *setup(int argc, char **argv) {
    image *photo;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <infile> <outfile>\n\n", argv[0]);
        return NULL;
    }

    photo = read_image(argv[1]);
    if (photo == NULL) {
        fprintf(stderr, "Unable to read input file %s\n\n", argv[1]);
        return NULL;
    }

    return photo;
}

void cleanup(image *photo, char **argv) {
    int rc = write_image(argv[2], photo);
    if (!rc) {
        fprintf(stderr, "Unable to write output file %s\n\n", argv[2]);
    }

    clear_image(photo);
}

int main(int argc, char **argv) {
    image *photo = setup(argc, argv);
    process_data(photo);
    cleanup(photo, argv);
    return 0;
}
