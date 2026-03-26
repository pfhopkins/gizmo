/* This is a generic code block designed for simple neighbor loops, so that they don't have to 
    be copy-pasted and can be generically optimized in a single place. specifically this is for
    the initial loop of particles on the local processor (and determination of which need passing) 

   Two blocks need to be defined or this will crash: 
   CONDITION_FOR_EVALUATION inserts the clause that actually determines
        whether or not to pass a particle to the main evaluation routine
   EVALUATION_CALL is the actual call, and needs to be written appropriately
 */
#if !defined(CONDITION_FOR_EVALUATION) || !defined(EVALUATION_CALL)
printf("Cannot compile the primary sub-loop without both CONDITION_FOR_EVALUATION and EVALUATION_CALL defined. Exiting. \n"); fflush(stdout); exit(995533);
#endif
/* variable assignment */
int i, j, *exportflag, *exportnodecount, *exportindex, *ngblist, thread_id = *(int *) p;
/* define the pointers needed for each thread to speak back regarding what needs processing */
ngblist = Ngblist.data() + thread_id * NumPart;
exportflag = Exportflag + thread_id * NTask;
exportnodecount = Exportnodecount + thread_id * NTask;
exportindex = Exportindex + thread_id * NTask;
/* Note: exportflag is local to each thread */
for(j = 0; j < NTask; j++) {exportflag[j] = -1;}
#ifdef _OPENMP
if(BufferCollisionFlag && thread_id) {return NULL;} /* force to serial for this subloop if threads simultaneously cross the Nexport bunchsize threshold */
#endif
/* now begin the actual loop, grabbing batches of particles to reduce critical section overhead */
#ifndef PRIMARY_LOOP_BATCH_SIZE
#define PRIMARY_LOOP_BATCH_SIZE 8
#endif
while(1)
{
    int batch[PRIMARY_LOOP_BATCH_SIZE], batch_count = 0;
#ifdef _OPENMP
#pragma omp critical(_nextlistprimblox_)
#endif
    {
        while(batch_count < PRIMARY_LOOP_BATCH_SIZE && BufferFullFlag == 0 && NextParticle < (int)ActiveParticleList.size())
        {
            int idx = ActiveParticleList[NextParticle];
            NextParticle++;
            if(!ProcessedFlag[idx]) {batch[batch_count++] = idx;}
        }
    }
    if(batch_count == 0) {break;}
    int buffer_full = 0;
    for(int b = 0; b < batch_count; b++)
    {
        i = batch[b];
        CONDITION_FOR_EVALUATION
        {
            if(EVALUATION_CALL < 0) {buffer_full = 1; break;} // export buffer has filled up //
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    if(buffer_full) {break;}
}
/* loop completed successfully */
return NULL;
