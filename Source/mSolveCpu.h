#ifndef _mSolveCpu_
#define _mSolveCpu_

#include "JSphCpu.h"

class mSolveCpu : public JSphCpu
{
	float *r;
	float *rBar;
	float *v;
	float *p;
	float *s;
	float *t;
	float *y;
	float *z;
	float *X;
	float *Xerror;

	void biCGStab(); //Linear Solver
	void pReorder();
	float l2norm(float *residual);
	float mMultiply(float *a, float *b);
public:
	mSolveCpu();
	~mSolveCpu();
};
#endif