// a stand-in for libnvidia-ml with fixed readings, for test_stats_history.py (LLAMA_SERVER_NVML_LIB): one GPU
#include <string.h>

typedef struct { unsigned int gpu; unsigned int memory; } util_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } mem_t;

static int dev;

int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int * n) { *n = 1; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int i, void ** d) { if (i != 0) return 2; *d = &dev; return 0; }
int nvmlDeviceGetName(void * d, char * name, unsigned int len) { (void) d; strncpy(name, "Fake GPU", len); return 0; }
int nvmlDeviceGetUtilizationRates(void * d, util_t * u) { (void) d; u->gpu = 42; u->memory = 7; return 0; }
int nvmlDeviceGetMemoryInfo(void * d, mem_t * m) { (void) d; m->total = 4ULL << 30; m->used = 1ULL << 30; m->free = 3ULL << 30; return 0; }
int nvmlDeviceGetPcieThroughput(void * d, int c, unsigned int * v) { (void) d; *v = c == 1 ? 1000 : 10; return 0; }
int nvmlDeviceGetTemperature(void * d, int s, unsigned int * v) { (void) d; (void) s; *v = 61; return 0; }
int nvmlDeviceGetPowerUsage(void * d, unsigned int * v) { (void) d; *v = 70500; return 0; }
int nvmlDeviceGetClockInfo(void * d, int t, unsigned int * v) { (void) d; *v = t == 1 ? 1905 : 0; return 0; }
