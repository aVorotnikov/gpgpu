#pragma once
#include "device_launch_parameters.h" 

struct particle;

// ���������� ��� �������
enum physics_type {
	SPACE,
	EARTH_PHYSICS
};

// ����������� ���
struct space_physics {
public:
	__device__ void affect(particle* p) {};
};

// ������ ���, ��������� ��������������� ����������
struct erath_physics {
public:
	__device__	void affect(particle* p);

private:
	float g = 9.8f;
};

// ����������� ������ ��� �������� � ������������ ���������� ����� �������
class physics_manager {
public:
	__device__ physics_manager() {};
	__device__ ~physics_manager() {};
	__device__	void physicsMakeAction(particle* p);

};
