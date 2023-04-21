var locNames = ['Alatna', 'Huslia', 'Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5', 'Loc6', 'Loc7', 'Loc8'];

var fireLocs = [[-153.65, 66.70],
                [-155.69, 66.01],
                [-157.01, 66.96],
                [-157.82, 65.81],
                [-158.34, 65.23],
                [-153.05, 67.12],
                [-152.30, 67.12],
                [-150.92, 66.71],
                [-146.44, 66.96],
                [-153.26, 64.53]];

// Convert the list of points to a feature collection
var points = ee.FeatureCollection(fireLocs.map(function(coord) {
  var lon = ee.Number(coord[0]);
  var lat = ee.Number(coord[1]);
  var point = ee.Geometry.Point(lon, lat);
  return ee.Feature(point);
}));

// Plot the points on the map
Map.addLayer(points, {color: 'red'}, 'Points');
//======================================================================================
//======================================================================================
// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
//  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 
//  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 
// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
//======================================================================================
//======================================================================================

var LOC = 0;
var idx = 195;

//======================================================================================
//======================================================================================
// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
//  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 
//  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 
// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
//======================================================================================
//======================================================================================
var lon = fireLocs[LOC][0];
var lat = fireLocs[LOC][1];

var min_lon = lon - .65;
var max_lon = lon + .65;
var min_lat = lat - .25;
var max_lat = lat + .25;

var rect = ee.Geometry.Polygon([
  [min_lon, min_lat],
  [max_lon, min_lat],
  [max_lon, max_lat],
  [min_lon, max_lat],
  [min_lon, min_lat]]);

Map.addLayer(rect, {}, 'Fire area');

var dataset = ee.ImageCollection('MODIS/061/MYD14A1')
                  .filter(ee.Filter.date('2016-01-01', '2016-12-31'))
                  .filterBounds(ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)); 

var fireMaskVis = {
  min: 0.0,
  max: 6000.0,
  bands: ['MaxFRP', 'FireMask', 'FireMask'],
};
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
var fireClipped = dataset.map(function(image) {
  return image.clip(ee.Geometry.Rectangle(min_lon, min_lat, max_lon, max_lat));
});

print('Type of fireClipped:', fireClipped);

var fireCollection = ee.ImageCollection(fireClipped);

print('Length of fireCollection:', fireCollection.size());

Map.setCenter(min_lon, min_lat, 7);

var point = ee.Geometry.Point(lon, lat);

// Add the point as a layer to the map
Map.addLayer(point, {color: 'red'}, 'My Point');
//======================================================================================
//======================================================================================
// Define a function to sum the values of the MaxFRP band
var sumMaxFRP = function(image) {
  // Extract the MaxFRP band
  var maxFRP = image.select('MaxFRP');
  // Compute the sum of the MaxFRP values
  var sum = maxFRP.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: image.geometry(),
    scale: 1000,
    maxPixels: 1e9
  }).get('MaxFRP');
  // Set the sum as the feature property and return the feature
  return ee.Feature(image.geometry(), {'sum': sum});
};

// Map the sumMaxFRP function over the fireCollection
var summedMaxFRP = fireCollection.map(sumMaxFRP);

var sumList = summedMaxFRP.aggregate_array('sum');

// Print the resulting list of sums
print('Sums:', sumList);
//======================================================================================
//======================================================================================
var fireClippedList = fireClipped.toList(9999);

var curr = fireClippedList.get(idx);
//======================================================================================
//======================================================================================
print("curr=", curr);

print("sum[0]", ee.Number(sumList.get(idx)));
var im = ee.Image(curr);
//===========================================================
//===========================================================
var curr_mask1 = im.select('FireMask');

var mask1_stats = curr_mask1.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: curr_mask1.geometry(),
});

var min_mask1 = mask1_stats.get('FireMask_min').getInfo();
var max_mask1 = mask1_stats.get('FireMask_max').getInfo();

//print('Minimum FireMask Value:', min_mask1);
//print('Maximum FireMask Value:', max_mask1);

var curr_mask1_Vis = {
  min: min_mask1,
  max: max_mask1,
  bands: ['FireMask'],
};
//===========================================================
//===========================================================
var curr_mask2 = im.select('MaxFRP');

var mask2_stats = curr_mask2.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: curr_mask2.geometry(),
});

var min_mask2 = mask2_stats.get('MaxFRP_min').getInfo();
var max_mask2 = mask2_stats.get('MaxFRP_max').getInfo();

//print('Minimum MaxFRP Value:', min_mask2);
//print('Maximum MaxFRP Value:', max_mask2);

var curr_mask2_Vis = {
  min: min_mask2,
  max: max_mask2,
  bands: ['MaxFRP'],
};
//===========================================================
//===========================================================
var curr_mask3 = im.select('sample');

var mask_stats = curr_mask3.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: curr_mask3.geometry(),
});

var min_mask3 = mask_stats.get('sample_min').getInfo();
var max_mask3 = mask_stats.get('sample_max').getInfo();

//print('Minimum sample Value:', min_mask3);
//print('Maximum sample Value:', max_mask3);

var curr_mask3_Vis = {
  min: min_mask3,
  max: max_mask3,
  bands: ['sample'],
};
//===========================================================
//===========================================================
var curr_mask4 = im.select('QA')

var mask_stats = curr_mask4.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: curr_mask4.geometry(),
});

var min_mask4 = mask_stats.get('QA_min').getInfo();
var max_mask4 = mask_stats.get('QA_max').getInfo();

//print('Minimum QA Value:', min_mask4);
//print('Maximum QA Value:', max_mask4);

var curr_mask4_Vis = {
  min: min_mask4,
  max: max_mask4,
  bands: ['QA'],
};
//===========================================================
//===========================================================
print("curr_mask=", curr_mask1);
print("im=", im.id());

//Map.addLayer(curr, snowCoverVis, 'Snow Cover')

Map.addLayer(im, fireMaskVis, 'Fire');

//Map.addLayer(curr_mask1, curr_mask1_Vis, 'FireMask');
Map.addLayer(curr_mask2, curr_mask2_Vis, 'Radiance');
//Map.addLayer(curr_mask3, curr_mask3_Vis, 'Sample');
//Map.addLayer(curr_mask4, curr_mask4_Vis, 'QA');
//===========================================================
//===========================================================
//===========================================================
//===========================================================
var trans = [30.00, 0.00,-2189805.00, 0.00,-30.00, 2395455.00];


Export.image.toDrive({
  image: curr_mask2,
  description: 'Modis_MaxFRP_'+locNames[LOC]+'_'+String(idx),
  crs: 'EPSG:3338',
  crsTransform: trans,
  region: ee.Geometry.Rectangle(min_lon, min_lat, max_lon, max_lat)
});