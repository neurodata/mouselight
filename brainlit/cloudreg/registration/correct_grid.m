function [corrected_image] = correct_grid(img, x_axis_coords, y_axis_coords, z_axis, grid_correction_blur_width)
    image_sum = sum(img,z_axis);

    % blur
    [Xgrid,Ygrid] = meshgrid(x_axis_coords,y_axis_coords);
    K = exp(-(Xgrid.^2 + Ygrid.^2)/2/(grid_correction_blur_width)^2);
    K = K / sum(K(:));
    Ks = ifftshift(K);
    Kshat = fftn(Ks);
    Jb = ifftn(fftn(image_sum).*Kshat,'symmetric');
    corrected_image = bsxfun(@times, img, Jb./(image_sum+1));
end