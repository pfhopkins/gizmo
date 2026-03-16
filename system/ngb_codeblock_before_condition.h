/*
 this defines a code-block to be inserted in the neighbor search routines after the conditions for neighbor-validity are applied
 (valid particle types checked)
 */
  int numngb, no, p, task;
  struct NODE *current;
  // cache some global vars locally for improved compiler alias analysis
  int maxPart = All.MaxPart;
  int maxNodes = MaxNodes;
  long bunchSize = All.BunchSize;
  integertime ti_Current = All.Ti_Current;
  MyDouble dx, dy, dz, dist, xtmp; xtmp=0;

  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < maxPart)		/* single particle */
	{
	  p = no;
	  no = Nextnode[no];
