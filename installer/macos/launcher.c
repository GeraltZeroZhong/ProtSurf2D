#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    char executable_path[PATH_MAX];
    uint32_t path_size = sizeof(executable_path);
    if (_NSGetExecutablePath(executable_path, &path_size) != 0) {
        fputs("TopoPPI application path is too long.\n", stderr);
        return 1;
    }

    char *filename = strrchr(executable_path, '/');
    if (filename == NULL) {
        fputs("TopoPPI application path is invalid.\n", stderr);
        return 1;
    }
    strcpy(filename + 1, "launch.sh");

    char **shell_argv = calloc((size_t)argc + 2, sizeof(char *));
    if (shell_argv == NULL) {
        fputs("TopoPPI could not allocate launcher arguments.\n", stderr);
        return 1;
    }
    shell_argv[0] = "/bin/bash";
    shell_argv[1] = executable_path;
    for (int index = 1; index < argc; ++index) {
        shell_argv[index + 1] = argv[index];
    }

    execv(shell_argv[0], shell_argv);
    perror("TopoPPI could not start");
    free(shell_argv);
    return 1;
}
