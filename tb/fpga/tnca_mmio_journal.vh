// Test-only timed mailbox snapshots. All file I/O consumes zero simulated
// time. Wall-clock overhead and its effect on live host delivery are real;
// replay removes that delivery variability for a captured experiment.
integer j_capture=0,j_replay=0,j_index=0,j_batches=0,j_clean_stop=0;
reg [8*512-1:0] j_dir,j_replay_dir,j_file;
reg [8*8-1:0] j_kind;
reg [63:0] j_time,j_tid;
integer j_n,j_poll,j_inter,j_watch;

task journal_copy;
 input [8*512-1:0] source_path,target_path;
 integer source_fd,target_fd,character;
 begin
  source_fd=$fopen(source_path,"r");target_fd=$fopen(target_path,"w");
  if(source_fd==0 || target_fd==0)$fatal(1,"journal file copy failed: %0s -> %0s",source_path,target_path);
  character=$fgetc(source_fd);
  while(character!=-1)begin $fwrite(target_fd,"%c",character);character=$fgetc(source_fd);end
  $fclose(source_fd);$fclose(target_fd);
 end
endtask

task journal_init;
 begin
  j_capture=$value$plusargs("journal=%s",j_dir);
  j_replay=$value$plusargs("replay=%s",j_replay_dir);
  if(j_capture && j_replay)$fatal(1,"journal and replay are mutually exclusive");
  if(j_capture)begin
   $sformat(j_file,"%0s/index.txt",j_dir);j_index=$fopen(j_file,"w");
   if(j_index==0)$fatal(1,"cannot open journal index");
   $fwrite(j_index,"H %0d %0d %0d\n",pollcycles,intercycle,watchdog);$fflush(j_index);
  end
 end
endtask

task journal_begin;
 input [63:0] tid;
 begin if(j_capture)begin
  $sformat(j_file,"%0s/%0d.cmd",j_dir,tid);journal_copy(cmd_path,j_file);
  $fwrite(j_index,"B %0d %0d\n",$time,tid);$fflush(j_index);
 end end
endtask

task journal_end;
 input [63:0] tid;
 integer response_fd,value_index;
 begin if(j_capture)begin
  $sformat(j_file,"%0s/%0d.rsp",j_dir,tid);response_fd=$fopen(j_file,"w");
  if(response_fd==0)$fatal(1,"cannot write journal response");
  $fwrite(response_fd,"%0d\n",tid);
  for(value_index=0;value_index<nvalues;value_index=value_index+1)
   $fwrite(response_fd,"%016x\n",values[value_index]);
  $fwrite(response_fd,"end\n");$fclose(response_fd);
  // Commit before publishing the response to the live host, which may close
  // its simulator immediately after receiving the final response.
  $fwrite(j_index,"E %0d %0d\n",$time,tid);$fflush(j_index);
 end end
endtask

task journal_compare_response;
 input [8*512-1:0] expected_path;
 integer expected_fd,actual_fd,expected_byte,actual_byte;
 begin
  expected_fd=$fopen(expected_path,"r");actual_fd=$fopen(rsp_path,"r");
  if(expected_fd==0 || actual_fd==0)$fatal(1,"missing replay response");
  expected_byte=$fgetc(expected_fd);actual_byte=$fgetc(actual_fd);
  while(expected_byte!=-1 && actual_byte!=-1)begin
   if(expected_byte!==actual_byte)$fatal(1,"replay response differs for batch %0d",last_tid);
   expected_byte=$fgetc(expected_fd);actual_byte=$fgetc(actual_fd);
  end
  if(expected_byte!==actual_byte)$fatal(1,"replay response length differs for batch %0d",last_tid);
  $fclose(expected_fd);$fclose(actual_fd);
 end
endtask

task journal_poll_stop;
 integer stop_fd;
 begin if(j_capture)begin
  $sformat(j_file,"%0s/stop.request",j_dir);stop_fd=$fopen(j_file,"r");
  if(stop_fd!=0)begin
   $fclose(stop_fd);j_clean_stop=1;$finish;
  end
 end end
endtask

// A failure may occur in an idle interval AFTER the last complete batch.
// Preserve that interval; an E record by itself is not a complete capture.
final begin
 if(j_capture && j_index!=0)begin
  $fwrite(j_index,"T %0d %0d\n",$time,j_clean_stop);$fflush(j_index);$fclose(j_index);
 end
end

task journal_replay_run;
 begin if(j_replay)begin
  $sformat(j_file,"%0s/index.txt",j_replay_dir);j_index=$fopen(j_file,"r");
  if(j_index==0)$fatal(1,"cannot read replay index");
  j_n=$fscanf(j_index,"%s %d %d %d",j_kind,j_poll,j_inter,j_watch);
  if(j_n!=4 || j_kind!="H" || j_poll<=0 || j_inter<0 || j_watch<=0)$fatal(1,"invalid replay header");
  intercycle=j_inter;
  j_n=$fscanf(j_index,"%s %d %d",j_kind,j_time,j_tid);
  while(j_n==3)begin
   if(j_kind=="T")begin
    if(j_time<$time || (j_tid!=0 && j_tid!=1))$fatal(1,"invalid replay stop marker");
    #(j_time-$time);
    // Allow all DUT active/NBA events at the captured failure edge to run.
    #1;
    if(j_tid==0)$fatal(1,"captured abnormal stop reached without expected DUT failure");
    if(j_batches==0)$fatal(1,"empty capture");
    j_n=$fgetc(j_index);
    while(j_n==9 || j_n==10 || j_n==13 || j_n==32)j_n=$fgetc(j_index);
    if(j_n!=-1)$fatal(1,"data after terminal replay marker");
    $fclose(j_index);
    $display("MMIO_JOURNAL_REPLAY_PASS batches=%0d time=%0t",j_batches,$time);
    $finish;
   end
   if(j_kind!="B" || j_time<$time)$fatal(1,"invalid/past replay batch time");
   #(j_time-$time);
   $sformat(cmd_path,"%0s/%0d.cmd",j_replay_dir,j_tid);
   run_batch(j_tid);
   j_n=$fscanf(j_index,"%s %d %d",j_kind,j_time,j_tid);
   if(j_n!=3 || j_kind!="E")$fatal(1,"partial captured batch returned without its original failure");
   if(j_tid!==last_tid || j_time!=$time)$fatal(1,"replay timing differs at batch %0d",last_tid);
   $sformat(j_file,"%0s/%0d.rsp",j_replay_dir,j_tid);journal_compare_response(j_file);
   j_batches=j_batches+1;
   j_n=$fscanf(j_index,"%s %d %d",j_kind,j_time,j_tid);
  end
  $fatal(1,"capture lacks terminal marker; complete replay cannot be certified");
 end end
endtask
