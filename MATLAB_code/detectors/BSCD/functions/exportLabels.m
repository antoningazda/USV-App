function exportLabels(labels, filePath)
    fid = fopen(filePath, 'w');
    for i = 1:length(labels)
        fprintf(fid, '%.6f\t%.6f\td\n', labels(i).StartTime, labels(i).EndTime);
        fprintf(fid, '\\\t0.000000\t0.000000\n');
    end
    fclose(fid);
end