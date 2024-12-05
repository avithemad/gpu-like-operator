#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 1024

// Function to merge two lines and write to the output file
void merge_lines(FILE *infile, FILE *outfile) {
    char line1[MAX_LINE_LENGTH], line2[MAX_LINE_LENGTH];

    // Read the first line
    if (fgets(line1, MAX_LINE_LENGTH, infile) != NULL) {
        // Read the second line
        if (fgets(line2, MAX_LINE_LENGTH, infile) != NULL) {
            // Remove newline character from both lines if present
            line1[strcspn(line1, "\n")] = '\0';
            line2[strcspn(line2, "\n")] = '\0';

            // Write both lines concatenated to the output file
            fprintf(outfile, "%s %s\n", line1, line2);
        } else {
            // If there is no second line, just write the first line
            line1[strcspn(line1, "\n")] = '\0';
            fprintf(outfile, "%s\n", line1);
        }
    }
}

int main(int argc, char *argv[]) {
    // Check if the input file name is provided
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    FILE *infile = fopen(argv[1], "r");
    if (infile == NULL) {
        perror("Error opening input file");
        return 1;
    }

    FILE *outfile = fopen(argv[2], "w");
    if (outfile == NULL) {
        perror("Error opening output file");
        fclose(infile);
        return 1;
    }

    // Process the file by merging lines
    while (!feof(infile)) {
        merge_lines(infile, outfile);
    }

    // Close the files
    fclose(infile);
    fclose(outfile);

    printf("Processing complete. Output written to %s\n", argv[2]);
    return 0;
}
