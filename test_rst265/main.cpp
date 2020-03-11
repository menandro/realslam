#include <rst265/T265.h>

int main() {
	T265* t265 = new T265();
	t265->initialize("852212110449");
	t265->run();
	return 0;
}