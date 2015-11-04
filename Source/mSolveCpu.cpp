#include "mSolveCpu.h"

mSolveCpu::mSolveCpu():JSphCpu(false)
{
	r = new float[Npbf]; rBar = new float[Npbf];
	v = new float[Npbf]; p = new float[Npbf];
	y = new float[Npbf]; s = new float[Npbf];
	t = new float[Npbf]; z = new float[Npbf];
	X = new float[Npbf]; Xerror = new float[Npbf];
}

mSolveCpu::~mSolveCpu()
{
	delete[] r; r = NULL; delete[] rBar; rBar = NULL;
	delete[] v; v = NULL; delete[] p; p = NULL;
	delete[] y; y = NULL; delete[] s; s = NULL;
	delete[] t; t = NULL; delete[] z; z = NULL;
	delete[] X; X = NULL; delete[] Xerror; Xerror = NULL;
}

void mSolveCpu::biCGStab()
{
	float rho_nM1 = 1.0;
	float alpha = 1.0;
	float omega = 1.0;
	float rho_n;
	float error = 0.0;
	float tol = 1e-7f;

	memset(p, 0, matrix1d);
	memset(v, 0, matrix1d);
	memset(Xerror, 0, matrix1d);

	float normb = l2norm(matrixB);
	
	if(normb == 0)
	{
		for(int i = 0; i < Np; i++) X[i] = 0.0;

		return;
	}

	for(int i = 0; i < Np; i++) X[i] = Pressc[order[i]];

	//r_0 = B - AX_0, X_0 = press_n-1
	memcpy(r, matrixB, matrix1d);

	for(int i = 0; i < Np; i++)
	{
		for(int j = 0; j < Np; j++)
		{
			float input = -matrixA[i * Np+ j] * X[j];
			float d = input - error;
			float f = r[i] + d;
			error = (f - r[i]) - d;
			r[i] = f;
		}
		error = 0.0;
		//cout << "r_0[" << i << "] = " << r[i] << "\n";
	}

	float normr = l2norm(r);

	if(normr / normb <= tol)
	{
		return;
	}

	//rbar = r
	memcpy(rBar, r, matrix1d);

	//MAIN LOOP
	for(int n = 1; n < 1e3; n++)
	{
		//rho_n = (rBar, r_n-1)
		rho_n = 0;

		for(int i = 0; i < Np; i++)
		{
			float input = rBar[i] * r[i];

			float d = input - error;
			float f = rho_n + d;
			error = (f - rho_n) - d;
			rho_n = f;
		}
		//cout << "rho_n = "<< rho_n << "\n";
 		error = 0.0;

		if(n == 1)
		{
			memcpy(p, r, matrix1d);
		}
		else
		{
			//beta = (rhon/rho_n-1)(alpha/omega_n-1)
			float beta = (rho_n / rho_nM1) * (alpha / omega);
			//cout << "beta = "<< beta << "\n";
			//p_n = r_n-1 + beta(p_n-1 - omega_n-1*v_n-1)
    		for(int i = 0; i < Np; i++)
			{
				p[i] = r[i] + beta * (p[i] - omega * v[i]);
				//cout << "p["<< i << "] = " << p[i] << "\n";
			}
		}

		//Solve y from Ky = p_n
 		for(int i = 0; i < Np; i++)
		{
			y[i] = p[i] / matrixA[i * (Np + 1)];
			//cout << "y["<< i << "] = " << y[i] << "\n";
		}

		//v_n = Ay
   		memset(v, 0, matrix1d);

		for(int i = 0; i < Np; i++)
		{
			error = 0.0;
			for(int j = 0; j < Np; j++)
			{
				float input = matrixA[i * Np + j] * y[j];

				float d = input - error;
				float f = v[i] + d;
				error = (f - v[i]) - d;
				v[i] = f;
			}
			//cout << "v["<< i << "] = " << v[i] << "\n";			
		}

		//alpha = rho_n / (rBar, v_n)
		float sum = 0.0;

		for(int i = 0; i < Np; i++)
		{
			float input = rBar[i] * v[i];

			float d = input - error;
			float f = sum + d;
			error = (f - sum) - d;
			sum = f;
		}

		error = 0.0;	
		alpha = rho_n / sum;
		//cout << "alpha = " << alpha << "\n";
		//s = r_n-1 - alpha * v_n
 		for(int i = 0; i < Np; i++)
		{
			s[i] = r[i] - alpha * v[i];
			//cout << "s["<< i << "] = " << s[i] << "\n";
		}
		
		float norms = l2norm(s);

		if(norms / normr < tol)
		{
			for(int i = 0; i < Np; i++) X[i] += alpha * y[i];
			std::cout << n << "\n";
			break;
		}

		//Solve z from Kz = s
 		for(int i = 0; i < Np; i++)
		{	
			float preCond = matrixA[i * (Np + 1)];
			z[i] = s[i] / preCond;
			//cout << "z["<< i << "] = " << z[i] << "\n";
		}
		
		//t = Az
		memset(t, 0, matrix1d);

		for(int i = 0; i < Np; i++)
		{
			for(int j = 0; j < Np; j++)
			{
				float input = matrixA[i * Np + j] * z[j];

				float d = input - error;
				float f = t[i] + d;
				error = (f - t[i]) - d;
				t[i] = f;
			}
			//cout << "t["<< i << "] = " << t[i] << "\n";
			error = 0.0;
		}

		//omega_n = (KM1*t,KM1*s)/(KM1*t,KM1*t)
		sum = 0.0;
		float sum2 = 0.0;
		float error2 = 0.0;

		for(int i = 0; i < Np; i++)
		{
			float input = t[i] * s[i] / matrixA[i * (Np + 1)];

			float d1 = input - error;
			float f1 = sum + d1;
			error = (f1 - sum) - d1;
			sum = f1;

			input = t[i] * t[i] / matrixA[i * (Np + 1)];

			float d2 = input - error2;
			float f2 = sum2 + d2;
			error2 = (f2 - sum2) - d2;
			sum2 = f2;
		}

		error = 0.0;
		omega = sum / sum2;
		//cout << "omega = " << omega << "\n";

		//Solve X_n = X_nM1 + alpha*y + omega_n*z
		for(int i = 0; i < Np; i++)
		{
			float input = alpha * y[i] + omega * z[i];
			float d = input - Xerror[i];
			float f = X[i] + d;
			Xerror[i] = (f - X[i]) - d;
			X[i] = f;
			//cout << "press["<< i << "] = " << X[i] << "\n";
		}

		//r_n = s - omega_n*t 
		//and check residual, l-2 norm
		sum = 0.0;

		for(int i = 0; i < Np; i++)
		{
			r[i] = s[i] - omega * t[i];
			//cout << "r["<< i << "] = " << r[i] << "\n";
		}

		normr = l2norm(r);
		float check = normr / normb;
		//cout << check << "\n";
		if(check < tol) 
		{
			std::cout << n << "\n";	
			break;
		}
		error = 0.0;
		rho_nM1 = rho_n;
	}
}

float mSolveCpu::l2norm(float *residual)
{
	float norm = 0.0;
	float error = 0.0;

	for(int i = 0; i < Np; i++)
	{
		float r2 = residual[i] * residual[i];

		float d = r2 - error;
		float f = norm + d;
		error = (f - norm) - d;
		norm = f;
	}
	
	norm = sqrtf(norm);

	return norm;
}

float mSolveCpu::mMultiply(float *a, float *b)
{
	float c = 0.0f;
	float error = 0.0f;

	for(int i = 0; i = Np; i++)
	{
		float add = a[i] * b[i];
		float d = add - error;
		float f = c + d;
		error = (f - c) - d;
		c = f;
	}

	return c;
}

void mSolveCpu::pReorder()
{
	for(int i = 0; i < Npbf; i++) press[order[i]] = X[i];

	for(int i = Npbf; i < Np; i++) Pressc[m] = Pressc[mfpn[i]] - rho * g * (Posc[i].z - Posc[i].z); 	
}