import numpy as np

def compute_position_from_binary_signals(A, B,
                                         perimeter_cm=25,
                                         cpr=1000):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.

    Algorithm based on the schematic of cases shown in the doc
    ---------------
    Input:
        A, B - traces to convert
   
    Output:
        Positions through time

    

    '''

    Delta_position = 0*A[:-1] # N-1 elements

    ################################
    ## positive_increment_cond #####
    ################################
    PIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[PIC] = +1

    ################################
    ## negative_increment_cond #####
    ################################
    NIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[NIC] = -1

    position = np.cumsum(np.concatenate([[0], Delta_position]))

    return position*perimeter_cm/cpr


if __name__=='__main__':

    N = 100
    A, B = np.random.randint(0, 2, size=(2,N))

    x = compute_position_from_binary_signals(A, B)

    from datavyz import ges as ge
    ge.plot(x, fig_args=dict(figsize=(2,1)))
    ge.show()
