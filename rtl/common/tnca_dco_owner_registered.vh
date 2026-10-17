// Per-module function template; deliberately has no global include guard.
    // Endpoints are registered atomically with BASE/SIZE at TMU DESC commit.
    function automatic [11:0] owner_registered;
        input [ADDR_W-1:0] a;
        input [DCO_T-1:0] tv;
        input [DCO_T*64-1:0] tbase;
        input [DCO_T*65-1:0] tend;
        input [DCO_T-1:0] tbyp;
        input [DCO_T*6-1:0] tlsb;
        integer k, j;
        reg [63:0] b;
        reg [DCO_T-1:0] eligible_mask;
        reg blocked;
        begin
            eligible_mask = {DCO_T{1'b0}};
            for (k = 0; k < DCO_T; k = k + 1) begin
                b = tbase[k*64 +: 64];
                if (tv[k] && (a >= b) &&
                    (tend[k*65+64] || (a < tend[k*65 +: 64])))
                    eligible_mask[k] = 1'b1;
            end
            owner_registered = 12'd0;
            if (|eligible_mask) begin
                for (k = 0; k < DCO_T; k = k + 1) begin
                    b = tbase[k*64 +: 64];
                    blocked = 1'b0;
                    for (j = 0; j < DCO_T; j = j + 1) begin
                        if (j < k) begin
                            if (eligible_mask[j] && tbase[j*64 +: 64] >= b) blocked = 1'b1;
                        end else if (j > k) begin
                            if (eligible_mask[j] && tbase[j*64 +: 64] > b) blocked = 1'b1;
                        end
                    end
                    if (eligible_mask[k] && !blocked)
                        owner_registered = owner_registered |
                            {1'b1, tbyp[k], k[3:0], tlsb[k*6 +: 6]};
                end
            end
        end
    endfunction

