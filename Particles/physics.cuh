#pragma once
#include "device_launch_parameters.h" 

struct particle;

// физический тип частицы
enum physics_type {
	SPACE,
	EARTH_PHYSICS
};

// космический тип
struct space_physics {
public:
	__device__ void affect(particle* p) {};
};

// земной тип, добавляем гравитационнную постоянную
struct erath_physics {
public:
	__device__	void affect(particle* p);

private:
	float g = 9.8f;
};

// определение класса для действия с определенным физическим типом частицы
class physics_manager {
public:
	__device__ physics_manager() {};
	__device__ ~physics_manager() {};
	__device__	void physicsMakeAction(particle* p);

};
