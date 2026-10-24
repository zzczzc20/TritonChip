// Test-only sparse backing for a large AXI window (R41 hbm8).
//
// A map file lists the backed window byte ranges, one "<offset> <bytes>" pair
// of hex numbers per line, 64-byte aligned, ascending and disjoint. Ranges
// occupy consecutive compact line slots in file order, so every instance that
// loads the same file agrees on every compact index (the shared-BO mirror and
// the DMA model index each other's arrays by it). A window line outside every
// range has no slot: callers $fatal on it rather than returning zero, which
// makes an access with a lost or an extra address bit a hard failure.
// Included inside a module; deliberately has no include guard.
 localparam SPARSE_MAX=512;
 reg [63:0] sparse_lo[0:SPARSE_MAX-1];
 reg [63:0] sparse_n[0:SPARSE_MAX-1];
 reg [63:0] sparse_at[0:SPARSE_MAX-1];
 integer sparse_count=0;
 reg [63:0] sparse_total=0;
 task sparse_load;
  input [8*512-1:0] path;
  integer fd,r;
  reg [63:0] off,bytes,prev_end;
  begin
   fd=$fopen(path,"r");
   if(fd==0)$fatal(1,"sparse map %0s cannot be opened",path);
   sparse_count=0;sparse_total=0;prev_end=0;
   while(!$feof(fd))begin
    r=$fscanf(fd,"%h %h\n",off,bytes);
    if(r==2)begin
     if(off[5:0]!=0 || bytes[5:0]!=0 || bytes==0 || sparse_count>=SPARSE_MAX ||
        (sparse_count>0 && off<prev_end))
      $fatal(1,"sparse map %0s: bad or overlapping range %h+%h",path,off,bytes);
     sparse_lo[sparse_count]=off>>6;sparse_n[sparse_count]=bytes>>6;
     sparse_at[sparse_count]=sparse_total;sparse_total=sparse_total+(bytes>>6);
     prev_end=off+bytes;sparse_count=sparse_count+1;
    end else if(r!=-1 && !$feof(fd))$fatal(1,"sparse map %0s: unreadable line",path);
   end
   $fclose(fd);
   if(sparse_count==0)$fatal(1,"sparse map %0s has no ranges",path);
  end
 endtask
 // Compact slot of window line `line`, or -1 when it is not backed.
 function integer sparse_index;
  input [63:0] line;
  integer k;
  begin
   sparse_index=-1;
   for(k=0;k<sparse_count;k=k+1)
    if(line>=sparse_lo[k] && line-sparse_lo[k]<sparse_n[k])sparse_index=sparse_at[k]+(line-sparse_lo[k]);
  end
 endfunction
