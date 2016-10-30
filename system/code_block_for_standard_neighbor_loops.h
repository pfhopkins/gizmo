/* This is a generic code block designed for simple neighbor loops, so that they don't have to 
    be copy-pasted and can be generically optimized in a single place */



/* allocate buffers to arrange communication */
int j, k, ngrp, ndone, ndone_flag, recvTask, place, save_NextParticle;
long long NTaskTimesNumPart, n_exported = 0;
NTaskTimesNumPart = maxThreads * NumPart;
Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
All.BunchSize = (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct DATA_IN_STRUCT) + sizeof(struct DATA_OUT_STRUCT) + sizemax(sizeof(struct DATA_IN_STRUCT),sizeof(struct DATA_OUT_STRUCT))));
DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

NextParticle = FirstActiveParticle;	/* begin with this index */
do
{
    BufferFullFlag = 0;
    Nexport = 0;
    save_NextParticle = NextParticle;
    for(j = 0; j < NTask; j++) {Send_count[j] = 0; Exportflag[j] = -1;}
    /* do local particles and prepare export list */
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int mainthreadid = omp_get_thread_num();
#else
        int mainthreadid = 0;
#endif
        EVALUATION_SUBROUTINE_CALL_PRIMARY;	/* do local particles and prepare export list */
    }
    if(BufferFullFlag)
    {
        int last_nextparticle = NextParticle;
        NextParticle = save_NextParticle;
        while(NextParticle >= 0)
        {
            if(NextParticle == last_nextparticle) break;
            if(ProcessedFlag[NextParticle] != 1) break;
            ProcessedFlag[NextParticle] = 2;
            NextParticle = NextActiveParticle[NextParticle];
        }
        if(NextParticle == save_NextParticle) {endrun(116609);} /* in this case, the buffer is too small to process even a single particle */
        int new_export = 0;
        for(j = 0, k = 0; j < Nexport; j++)
        {
            if(ProcessedFlag[DataIndexTable[j].Index] != 2)
            {
                if(k < j + 1) {k = j + 1;}
                for(; k < Nexport; k++)
                {
                    if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                    {
                        int old_index = DataIndexTable[j].Index;
                        DataIndexTable[j] = DataIndexTable[k];
                        DataNodeList[j] = DataNodeList[k];
                        DataIndexTable[j].IndexGet = j;
                        new_export++;
                        DataIndexTable[k].Index = old_index;
                        k++;
                        break;
                    }
                }
            } else {new_export++;}
        }
        Nexport = new_export;
    }
    n_exported += Nexport;
    
    for(j = 0; j < NTask; j++) {Send_count[j] = 0;}
    for(j = 0; j < Nexport; j++) {Send_count[DataIndexTable[j].Task]++;}
    MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
        Nimport += Recv_count[j];
        if(j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }
    DATA_GET_VEC = (struct DATA_IN_STRUCT *) mymalloc("rt_cg_DataGet", Nimport * sizeof(struct DATA_IN_STRUCT));
    DATA_IN_VEC = (struct DATA_IN_STRUCT *) mymalloc("rt_cg_DataIn", Nexport * sizeof(struct DATA_IN_STRUCT));
    /* prepare particle data for export */
    for(j = 0; j < Nexport; j++)
    {
        place = DataIndexTable[j].Index;
        particle2in_rt_cg(&DATA_IN_VEC[j], place);
        memcpy(DATA_IN_VEC[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
    }
    /* exchange particle data */
    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;
        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the particles */
                MPI_Sendrecv(&DATA_IN_VEC[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct DATA_IN_STRUCT), MPI_BYTE, recvTask, TAG_RT_A,
                             &DATA_GET_VEC[Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(struct DATA_IN_STRUCT), MPI_BYTE, recvTask, TAG_RT_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    myfree(DATA_IN_VEC);
    DATA_RESULT_VEC = (struct DATA_OUT_STRUCT *) mymalloc("rt_cg_DataResult", Nimport * sizeof(struct DATA_OUT_STRUCT));
    DATA_OUT_VEC = (struct DATA_OUT_STRUCT *) mymalloc("rt_cg_DataOut", Nexport * sizeof(struct DATA_OUT_STRUCT));
    /* now do the particles that were sent to us */
    NextJ = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int mainthreadid = omp_get_thread_num();
#else
        int mainthreadid = 0;
#endif
        EVALUATION_SUBROUTINE_CALL_SECONDARY;
    }
    if(NextParticle < 0) {ndone_flag = 1;} else {ndone_flag = 0;}
    MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    /* get the result */
    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;
        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* send the results */
                MPI_Sendrecv(&DATA_RESULT_VEC[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(struct DATA_OUT_STRUCT), MPI_BYTE, recvTask, TAG_RT_B,
                             &DATA_OUT_VEC[Send_offset[recvTask]],
                             Send_count[recvTask] * sizeof(struct DATA_OUT_STRUCT), MPI_BYTE, recvTask, TAG_RT_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    
    /* add the result to the local particles */
    for(j = 0; j < Nexport; j++)
    {
        place = DataIndexTable[j].Index;
        for(k = 0; k < N_RT_FREQ_BINS; k++)
        {
            matrixmult_out[k][place] += DATA_OUT_VEC[j].matrixmult_out[k];
            matrixmult_sum[k][place] += DATA_OUT_VEC[j].matrixmult_sum[k];
        }
    }
    myfree(DATA_OUT_VEC);
    myfree(DATA_RESULT_VEC);
    myfree(DATA_GET_VEC);
}
while(ndone < NTask);

/* do final operations on results */
double dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval / All.cf_hubble_a;
int i;
for(i = 0; i < N_gas; i++)
if(P[i].Type == 0)
{
    for(k = 0; k < N_RT_FREQ_BINS; k++)
    {
        double fac_i = dt * rt_absorption_rate(i,k);
        if((1 + fac_i + matrixmult_sum[k][i]) < 0)
        {
            printf("1 + matrixmult_sum + rate= %g   matrixmult_sum=%g rate=%g i =%d\n", 1 + fac_i + matrixmult_sum[k][i], matrixmult_sum[k][i], fac_i, i);
            endrun(11111111);
        }
        /* the "1" here accounts for the fact that we must start from the previous photon number (the matrix includes only the "dt" term);
         the fac_i term here accounts for sinks [here, the rate of photon absorption]; the in*sum part below accounts for the re-arrangement
         of indices [swapping indices i and j in the relevant equations so we account for both sides of the difference terms */
        matrixmult_sum[k][i] += 1.0 + fac_i;
        matrixmult_out[k][i] += matrixmult_in[k][i] * matrixmult_sum[k][i];
    }
}
/* free memory */
myfree(DataNodeList);
myfree(DataIndexTable);
myfree(Ngblist);
