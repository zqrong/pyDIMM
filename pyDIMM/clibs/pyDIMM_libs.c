// Compile it by `gcc -lm -shared -fPIC -o pyDIMM_libs.so pyDIMM_libs.c`, and then you can import the .so file by ctypes in Python.
// @author Ziqi Rong <rongziqi@sjtu.edu.cn>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

double lgamma_0(double x)
{
    if (x < 1e-154)
        return lgamma(1e-154);//Avoid the case that lgamma(0) = -inf.
    else
        return lgamma(x);
}

double sum_alpha_calc(double **alpha, int k, int n_genes)
{
    double sum = 0;
    for (int i = 0; i < n_genes; i++)
    {
        sum += alpha[i][k];
    }
    return sum;
}

double sum_UMI_cell(double **ob_data, int j, int n_genes)
{
    double sum = 0;
    for (int i = 0; i < n_genes; i++)
    {
        sum += ob_data[i][j];
    }
    return sum;
}

double log_prob_term1(double **ob_data, double **alpha, int n_genes, int j, int k)
{
    double logproduct = 0;
    for (int i = 0; i < n_genes; ++i)
    {
        logproduct += (lgamma_0(ob_data[i][j] + alpha[i][k]) - lgamma_0(alpha[i][k]));
    }
    return logproduct;
}

double log_prob_term2(double *sum_alpha, double *T, int j, int k)
{
    return (lgamma_0(sum_alpha[k]) - lgamma_0(T[j] + sum_alpha[k]));
}

void EM(bool save_log, FILE *logFilePointer, int model_size, int n_cells, int n_genes, double **ob_data, double **alpha, double *pie, int max_iter, double max_pie_tol, double max_loglik_tol, double max_alpha_tol, double *res_vec, double **delta)
{
    if (max_iter == 0) return;
    //Pre-calculate.
    double *sum_alpha = (double*)calloc(model_size, sizeof(double));
    for (int k = 0; k < model_size; ++k)
    {
        sum_alpha[k] = sum_alpha_calc(alpha, k, n_genes);
    }
    double *T = (double*)calloc(n_cells, sizeof(double));
    for (int j = 0; j < n_cells; ++j)
    {
        T[j] = sum_UMI_cell(ob_data, j, n_genes);//T_j is the total number of unique UMIs for the t_th cell.
    }
    //E-step, update delta.
    for (int j = 0; j < n_cells; ++j)
    {
        double *log_denominator = (double*)calloc(model_size, sizeof(double));
        for (int k = 0; k < model_size; ++k)
        {
            log_denominator[k] = log_prob_term1(ob_data, alpha, n_genes, j, k) + log_prob_term2(sum_alpha, T, j, k) + log(pie[k]);
        }
        for (int k = 0; k < model_size; ++k)
        {
            double denominator = 0;
            for (int kk = 0; kk < model_size; ++kk)
            {
                denominator += exp(log_denominator[kk] - log_denominator[k]);
            }
            delta[j][k] = 1/denominator;
            // printf("delta[%d][%d]=%lf\n", j, k, delta[j][k]);
        }
        free (log_denominator);
    }
    //M-step, update pie.
    double *new_pie = (double*)calloc(model_size, sizeof(double));
    for (int k = 0; k < model_size; ++k)
    {
        double sum_delta = 0;
        for (int j = 0; j < n_cells; ++j)
        {
            sum_delta += delta[j][k];
        }
        new_pie[k] = sum_delta/n_cells;
        // printf("%lf ", sum_delta);
    }
    //M-step, update alpha.
    double **new_alpha = (double**)calloc(n_genes, sizeof(double*));
    for (int i = 0; i < n_genes; ++i)
    {
        new_alpha[i] = (double*)calloc(model_size, sizeof(double));
    }
    for (int i = 0; i < n_genes; ++i)
    {
        for (int k = 0; k < model_size; ++k)
        {
            double numerator = 0;
            double denominator = 0;
            for (int j = 0; j < n_cells; ++j)
            {
                numerator += delta[j][k] * ob_data[i][j]/(ob_data[i][j]-1+alpha[i][k]);
                denominator += delta[j][k] * T[j]/(T[j]-1+sum_alpha[k]);
            }
            if (numerator == denominator)//Deal with the case that numerator == denominator == 0
            {
                numerator = 1;
                denominator = 1;
            }
            new_alpha[i][k] = alpha[i][k] * numerator/denominator;
            if (new_alpha[i][k] < 1e-323)
                new_alpha[i][k] = 1e-323;
            // printf("a[%d][%d] *= %lf/%lf\n", i, k, numerator, denominator);
        }
    }
    //Calculate the pie tolerance.
    double pie_tol = 0;
    for (int k = 0; k < model_size; ++k)
    {
        pie_tol += ((new_pie[k] - pie[k])*(new_pie[k] - pie[k]));
    }
    pie_tol = sqrt(pie_tol);
    //Calculate the alpha tolerance.
    double alpha_tol = 0;
    for (int i = 0; i < n_genes; ++i)
    {
        for (int k = 0; k < model_size; ++k)
        {
            alpha_tol += ((new_alpha[i][k] - alpha[i][k])*(new_alpha[i][k] - alpha[i][k]));
        }
    }
    alpha_tol = sqrt(alpha_tol);
    //Calculate loglik, AIC, BIC.
    double loglik, AIC, BIC;
    loglik = 0;
    AIC = 0;
    BIC = 0;
    for (int k = 0; k < model_size; ++k)
    {
        sum_alpha[k] = sum_alpha_calc(new_alpha, k, n_genes);//Calculate sum_new_alpha.
    }
    for (int j = 0; j < n_cells; ++j)
    {
        double max_delta = 0;
        int max_k = 0;
        for (int k = 0; k < model_size; ++k)
        {
            if (delta[j][k] > max_delta) max_k = k;//max_k is the indicator function (I(Z_j = k)). i.e., the most probably model where the j_th cell comes from.
        }
        loglik += (log_prob_term1(ob_data, new_alpha, n_genes, j, max_k) + log_prob_term2(sum_alpha, T, j, max_k));
        // printf("loglik[%d][%d] = %lf", j, max_k, loglik);
    }
    AIC = (-2)*loglik + 2*((model_size*n_genes)+(model_size*n_cells));
    BIC = (-2)*loglik + log(n_genes*n_cells)*((model_size*n_genes)+(model_size*n_cells));
    //Calculate the loglik tolerance.
    double loglik_tol = fabs(loglik - res_vec[0]);
    //Store loglik, AIC, BIC to resvec.
    res_vec[0] = loglik;
    res_vec[1] = AIC;
    res_vec[2] = BIC;
    //Store the updated value and free the dynamic memory.
    for (int k = 0; k < model_size; ++k)
    {
        pie[k] = new_pie[k];
    }
    free(new_pie);
    double alpha_sum = 0;
    for (int i = 0; i < n_genes; ++i)
    {
        for (int k = 0; k < model_size; ++k)
        {
            alpha[i][k] = new_alpha[i][k];
            alpha_sum += alpha[i][k];
        }
        free(new_alpha[i]);
    }
    free(new_alpha);
    free(sum_alpha);
    free(T);
    //Print to log file.
    if (save_log) fprintf(logFilePointer, "%d,%lf,%lf,%lf,%lf,%lf\n", max_iter, pie_tol, loglik_tol, alpha_tol, loglik, alpha_sum);
    //Judge whether to recur EM.
    if ((max_iter > 1) && (pie_tol >= max_pie_tol) && (loglik_tol >= max_loglik_tol) && (alpha_tol >= max_alpha_tol))
    {
        // printf("EM again.\n");
        EM(save_log, logFilePointer, model_size, n_cells, n_genes, ob_data, alpha, pie, max_iter-1, max_pie_tol, max_loglik_tol, max_alpha_tol, res_vec, delta);
    }
    return;
}

//The entry for Python calling. Take the 1-D array for ob_data and alpha arguments, reshape them and then call EM function.
void EM_with_1dArr(int model_size, int n_cells, int n_genes, double *ob_data_1d, double *alpha_1d, double *pie, int max_iter, double max_pie_tol, double max_loglik_tol, double max_alpha_tol, double *res_vec, double *delta_1d, bool save_log)
{
    //Reshape obdata and alpha to 2-D array according to the n_genes and n_cells.
    double **ob_data = (double**)calloc(n_genes, sizeof(double*));
    for (int i = 0; i < n_genes; i++)
    {
        ob_data[i] = (double*)calloc(n_cells, sizeof(double));
        for (int j = 0; j < n_cells; j++)
        {
            ob_data[i][j] = ob_data_1d[j * n_genes + i];
            // printf("ob_data[%d][%d] = %lf\n", i, j, ob_data[i][j]);
        }
    }
    double **alpha = (double**)calloc(n_genes, sizeof(double*));
    for (int i = 0; i < n_genes; i++)
    {
        alpha[i] = (double*)calloc(model_size, sizeof(double));
        for (int k = 0; k < model_size; k++)
        {
            alpha[i][k] = alpha_1d[k * n_genes + i];
        }
    }
    //Create delta matrix.
    double **delta = (double**)calloc(n_cells, sizeof(double*));//The latent variable. delta[j][k] shows the probability that j_th cell is sampled from the k_th model.
    for (int j = 0; j < n_cells; ++j)
    {
        delta[j] = (double*)calloc(model_size, sizeof(double));
    }
    //Create Log file to record the EM log, and then call EM.
    FILE *logFilePointer;
    if (save_log)
    {
        logFilePointer = fopen("pyDIMM.log", "w");
        fprintf(logFilePointer, "remain_iter,pie_delta,loglik_delta,alpha_l2norm_delta,loglik,alpha_sum\n");
    }
    else
    {
        logFilePointer = NULL;
    }
    EM(save_log, logFilePointer, model_size, n_cells, n_genes, ob_data, alpha, pie, max_iter, max_pie_tol, max_loglik_tol, max_alpha_tol, res_vec, delta);
    if (save_log)
    {
        fclose(logFilePointer);
    }
    //Store the value back to the 1-D array, so that the values can be retrieved by Python via pointer.
    for (int i = 0; i < n_genes; ++i)
    {
        for (int k = 0; k < model_size; ++k)
        {
             alpha_1d[k * n_genes + i] = alpha[i][k];
        }
    }
    for (int k = 0; k < model_size; ++k)
    {
        for (int j = 0; j < n_cells; ++j)
        {
            delta_1d[k * n_cells + j] = delta[j][k];
        }
    }
    //Free the dynamic variable.
    for (int i = 0; i < n_genes; i++)
    {
        free(ob_data[i]);
        free(alpha[i]);
    }
    free(ob_data);
    free(alpha);
    for (int j = 0; j < n_cells; ++j)
    {
        free(delta[j]);
    }
    free(delta);
}

void predict(int model_size, int n_cells, int n_genes, double **data, double **alpha, double *pie, double **delta, double **log_prob)
{
    double *sum_alpha = (double*)calloc(model_size, sizeof(double));
    for (int k = 0; k < model_size; ++k)
    {
        sum_alpha[k] = sum_alpha_calc(alpha, k, n_genes);
    }
    double *T = (double*)calloc(n_cells, sizeof(double));
    for (int j = 0; j < n_cells; ++j)
    {
        T[j] = sum_UMI_cell(data, j, n_genes);//T_j is the total number of unique UMIs for the t_th cell.
    }
    for (int j = 0; j < n_cells; ++j)
    {
        double *log_denominator = (double*)calloc(model_size, sizeof(double));
        for (int k = 0; k < model_size; ++k)
        {
            log_prob[j][k] = log_prob_term1(data, alpha, n_genes, j, k) + log_prob_term2(sum_alpha, T, j, k);
            log_denominator[k] = log_prob[j][k] + log(pie[k]);
        }
        for (int k = 0; k < model_size; ++k)
        {
            double denominator = 0;
            for (int kk = 0; kk < model_size; ++kk)
            {
                denominator += exp(log_denominator[kk] - log_denominator[k]);
            }
            delta[j][k] = 1/denominator;
            // printf("delta[%d][%d]=%lf\n", j, k, delta[j][k]);
        }
        free (log_denominator);
    }
    free(sum_alpha);
    free(T);
    return;
}

void predict_with_1dArr(int model_size, int n_cells, int n_genes, double *data_1d, double *alpha_1d, double *pie, double *delta_1d, double *log_prob_1d)
{
    //Reshape obdata and alpha to 2-D array according to the n_genes and n_cells.
    double **data = (double**)calloc(n_genes, sizeof(double*));
    for (int i = 0; i < n_genes; i++)
    {
        data[i] = (double*)calloc(n_cells, sizeof(double));
        for (int j = 0; j < n_cells; j++)
        {
            data[i][j] = data_1d[j * n_genes + i];
            // printf("data[%d][%d] = %lf\n", i, j, data[i][j]);
        }
    }
    double **alpha = (double**)calloc(n_genes, sizeof(double*));
    for (int i = 0; i < n_genes; i++)
    {
        alpha[i] = (double*)calloc(model_size, sizeof(double));
        for (int k = 0; k < model_size; k++)
        {
            alpha[i][k] = alpha_1d[k * n_genes + i];
            // printf("alpha[%d][%d]=%lf\n", i, k, alpha[i][k]);
        }
    }
    //Create delta matrix.
    double **delta = (double**)calloc(n_cells, sizeof(double*));//The latent variable. delta[j][k] shows the probability that j_th cell is sampled from the k_th model.
    for (int j = 0; j < n_cells; ++j)
    {
        delta[j] = (double*)calloc(model_size, sizeof(double));
    }
    //Create log_prob matrix.
    double **log_prob = (double**)calloc(n_cells, sizeof(double*));
    for (int j = 0; j < n_cells; ++j)
    {
        log_prob[j] = (double*)calloc(model_size, sizeof(double));
    }
    predict(model_size, n_cells, n_genes, data, alpha, pie, delta, log_prob);
    //Store delta matrix back.
    for (int k = 0; k < model_size; ++k)
    {
        for (int j = 0; j < n_cells; ++j)
        {
            delta_1d[k * n_cells + j] = delta[j][k];
        }
    }
    //Store delta matrix back.
    for (int k = 0; k < model_size; ++k)
    {
        for (int j = 0; j < n_cells; ++j)
        {
            log_prob_1d[k * n_cells + j] = log_prob[j][k];
        }
    }
    //Free the dynamic variable.
    for (int j = 0; j < n_cells; ++j)
    {
        free(delta[j]);
    }
    free(delta);
    for (int j = 0; j < n_cells; ++j)
    {
        free(log_prob[j]);
    }
    free(log_prob);
    for (int i = 0; i < n_genes; i++)
    {
        free(data[i]);
        free(alpha[i]);
    }
    free(data);
    free(alpha);
}
