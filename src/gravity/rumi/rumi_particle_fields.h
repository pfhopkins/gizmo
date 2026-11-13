/* RUMI++ per-particle flags; included inside struct particle_data. Step state sits in the star record (rumi_member_state). */
    short int RumiIntegrated;    /*!< 1 while the star is on a RUMI++ step */
    short int RumiMember;        /*!< 1 once integrated by RUMI++ (joined the members' common bin) */
