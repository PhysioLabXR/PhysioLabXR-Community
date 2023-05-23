function buffer =  stream_in(fn, only_stream, ignore_stream)
    if ~exist('only_stream','var')
        only_stream = [];
    end

    if ~exist('ignore_stream','var')
        ignore_stream = [];
    end

    % define constant 
    % magic =[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
    
    hex_string = '000102030405060708090A0B0C0D0E0F';
    magic = hex2dec(reshape(hex_string, 2, []).');
    len_magic  = length(magic);
    
    max_label_len = 32;
    max_dtype_len = 8;
    dim_bytes_len = 8;
    shape_bytes_len = 8;
    
    endianness = 'little';
    encoding = 'utf-8';
    
    ts_dtype = 'float64';
    
    
    % Specify the file path
    file_path = fn;

    %%
% Get the file information using the dir function
file_info = dir(file_path);

% Extract the size of the file from the file_info structure

total_bytes = file_info.bytes;
buffer = {};
read_bytes_count = 0;


% Open the file for reading
file_id = fopen(file_path, 'rb');

if file_id ~= -1
    while true

    % Read magic
    read_bytes = fread(file_id, 16,'uint8');
    read_bytes_count = read_bytes_count + length(read_bytes);
    
    if isempty(read_bytes)   
    break;
    end

    try
    assert(isequal(read_bytes, magic));
    catch ME
    if strcmp(ME.identifier, 'MATLAB:assertion:failed')
        error('Data invalid, magic sequence not found');
    else
        rethrow(ME);
    end
    end

   % read stream_label
   read_bytes = fread(file_id, max_label_len,'uint8');
   read_bytes_count = read_bytes_count + length(read_bytes);
   stream_name = strtrim(string((native2unicode(read_bytes, encoding).')));
   stream_name = strrep(stream_name, '.', '');

   % read read_bytes
   read_bytes = fread(file_id, max_dtype_len,'uint8');
   read_bytes_count = read_bytes_count + length(read_bytes);
   stream_dtype = strtrim(string((native2unicode(read_bytes, encoding).')));

    % read number of dimensions
   read_bytes = fread(file_id, dim_bytes_len,'uint8');
   read_bytes_count = read_bytes_count + length(read_bytes); 
   read_bytes = typecast(uint8(read_bytes), 'uint64');
   dims = read_bytes(1);      %% Read little?

   % read number of np shape
   shape = [];

   for i = 1:length(dims)+1
       read_bytes = fread(file_id, shape_bytes_len,'uint8');
       read_bytes_count = read_bytes_count + length(read_bytes); 
       read_bytes = typecast(uint8(read_bytes), 'uint64');
       shape(end+1) = read_bytes(1); 
   end

   if strcmp(stream_dtype, "float64")
    stream_dtype = 'double';
    end

   data_array_num_bytes = prod(shape) * numel(typecast(0, 'uint8')); %% need to varify
   timestamp_array_num_bytes = shape(end) * numel(typecast(0, 'uint8'));

 
    if isempty(only_stream)
    this_in_only_stream = true;
    else
    this_in_only_stream = ismember(stream_name, only_stream);
    end

  
    if isempty(ignore_stream)
    not_ignore_this_stream = true;
    else
    not_ignore_this_stream = ismember(stream_name, ignore_stream);
    end


    if not_ignore_this_stream && this_in_only_stream

    % read data array
   read_bytes = fread(file_id, data_array_num_bytes./8,'double');
   read_bytes_count = read_bytes_count + length(read_bytes);
 
   

   shape_num = prod(shape);
   data_array = read_bytes(1:shape_num);
    
    data_array = reshape(data_array, shape);
    
    % read timestamp array

    read_bytes = fread(file_id, timestamp_array_num_bytes./8,'double');
   ts_array = read_bytes;

    
    if  ~isfield(buffer, stream_name)
    buffer.(stream_name) = {zeros([shape(1:end-1), 0]), zeros([0, 1])}; % data first, timestamps second
  
    end
    
    

   buffer.(stream_name){1} = [buffer.(stream_name){1},data_array];
   buffer.(stream_name){2} = [buffer.(stream_name){2}; ts_array];

    else
    fread(file_id, data_array_num_bytes./8 + timestamp_array_num_bytes./8,'uint8');
   read_bytes_count = read_bytes_count + length(read_bytes);
    
    end
    end

    % Close the file
    fclose(file_id);
else
    disp('Failed to open the file.');
end

end