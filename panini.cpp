 
kernel PaniniProjection : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom, eEdgeClamped> src;
  Image<eWrite> dst;

  param:
    float FoV;
    float eye;
    float pan;
    float vsh;
   
    // filtering
    int filter;

  local:
    // universal constants
    float pi;
   
    // panini
    float Sppr;
    float d;
    float wfov;
    float Drpp;
   
    // image related
    float width;
    float height;
    float W;
   
    // filtering related
    bool rcomp;
    float weights[512];
    int wlut_size;
    int n; // filter size
    bool range_compress;


  // In define(), parameters can be given labels and default values.
  void define() {
    defineParam(FoV, "FoV", 150.0f);
    defineParam(eye, "eye", 1.0f);
    defineParam(pan, "pan", 0.0f);
    defineParam(vsh, "vsh", 0.0f);
  }

  // The init() function is run before any calls to process().
  // Local variables can be initialized here.
  void init() {
    pi = 3.1415926f;
    width = src.bounds.width();
    height = src.bounds.height();
    W = 2.0f;
    Sppr = W / (2.0f * pi); // source pixels / radians
    d = eye + 1.0f;
    wfov = (pi * min(FoV, 160.0f * d)) / 180.0f; //radians
    Drpp = 2.0f * d * tan(wfov / (2.0f * d)) / W; // destination coords in radians
   
    // filtering
    // Set filter size based on filter parameter
    // size of pixel neighborhood = n*2+1 by n*2+1
    n = 2; // default size is 3x3 except the following...
    if (filter == 0) n = 1; // Blackman-Harris = 2x2
    if (filter == 3) n = 1; // Cubic = 2x2
    if (filter == 2) n = 3; // Lanczos6 = 7x7
    if (filter == 9) n = 1; // Gaussian4

    rcomp = false;
    // Pre-calculate interpolation weights and store them in a lookup table
    wlut_size = 512; // Needs to match size of weights[] array
    // Loop over all indices of weights
    for (int i=0; i < wlut_size; i++) {
      // Calc weight for position at weight
      float x = float(i) * float(n) / float(wlut_size);
      weights[i] = weight(x);
    }
  }

  // Logarithmically compress values above 0.18. Matches OpenImageIO oiiotool rangecompress
  // Using log2shaper as this function can still cause overshoots
  float rangecompress(float x, bool inverse) {
    float x1 = 0.18f;
    float a = -0.54576885700225830078f;
    float b = 0.18351669609546661377f;
    float c = 284.3577880859375f;
    float absx = fabs(x);
    if (!inverse) {
      return absx <= x1 ? x : (a + b * log(fabs(c * absx + 1.0f))) * sign(x);
    } else {
      if (absx <= x1) return x;
      float e = exp((absx - a) / b);
      float _x = (e - 1.0f) / c;
      if (_x < x1) _x = (-e - 1.0f) / c;
      return _x * sign(x);
    }
  }

  // ACEScct log transform
  float acescct(float x, bool inverse) {
    if (inverse) {
      if (x > 0.155251141552511f) {
        return pow( 2.0f, x*17.52f - 9.72f);
      } else {
        return (x - 0.0729055341958355f) / 10.5402377416545f;
      }
    } else {
      if (x <= 0.0078125f) {
        return 10.5402377416545f * x + 0.0729055341958355f;
      } else {
        return (log2(x) + 9.72f) / 17.52f;
      }
    }
  }

  // Generic Log2 shaper
  float log2shaper(float x, bool inverse) {
    float max_exp = 2.0f;
    float min_exp = -2.0f;
    float mid_grey = 0.18f;
    float cut = 0.008;
    float slope = 1.0f/(log(2)*cut*(max_exp-min_exp));
    float offset = (log(cut/mid_grey)/log(2)-min_exp)/(max_exp-min_exp);
    if (inverse) {
      return x >= offset ? pow(2.0f, x*(max_exp-min_exp)+min_exp)*mid_grey : (x-offset)/slope+cut;
    } else {
      return x >= cut ? (log(x/mid_grey)/log(2)-min_exp)/(max_exp-min_exp) : slope*(x-cut)+offset;
    }
  }
 
  // Bilinear Interpolation
  float bilinear_filter(float x, float y, int k) {
    int u = floor(x);
    int v = floor(y);
    float a = x-u;
    float b = y-v;
    float p1 = src(u, v, k);
    float p2 = src(u+1, v, k);
    float p3 = src(u, v+1, k);
    float p4 = src(u+1, v+1, k);
    float r1 = p1 + a*(p2-p1);
    float r2 = p3 + a*(p4-p3);
    float r3 = r1 + b*(r2-r1);
    return r3;
  }


  // Gaussian interpolation https://renderman.pixar.com/resources/RenderMan_20/risOptions.html
  float gaussian(float x, float w) {
    x = fabs(x);
    float xw = 6*x/w;
    return x < w ? exp(-2*xw*xw) : 0.0f;
  }

  // Parameterized Cubic spline interpolation - https://www.desmos.com/calculator/il0lu3cnxr
  float bicubic(float x, float a, float b, float n) {
    x = fabs(x);
    if (x > n) return 0.0f;
    float x2 = x*x;
    float x3 = x*x*x;
    return x < 1.0f ? ((-6*a-9*b+12)*x3+(6*a+12*b-18)*x2-2*b+6)/6 : x < 2.0f ? ((-6*a-b)*x3+(30*a+6*b)*x2+(-48*a-12*b)*x+24*a+8*b)/6 : 0.0f;
  }

  // Catmull-Rom interpolation. Same as Keys (not used but included for posterity)
  float catrom(float x) {
    x        = fabs(x);
    float x2 = x * x;
    float x3 = x * x2;
    return (x >= 2.0f) ? 0.0f
                       : ((x < 1.0f) ? (3.0f * x3 - 5.0f * x2 + 2.0f)/2.0f
                       : (-x3 + 5.0f * x2 - 8.0f * x + 4.0f)/2.0f);
  }

  // Lanczos windowed sinc interpolation.
  // Lanczos4 a=2, Lanczos6 a=3
  float lanczos(float x, float a) {
    x = fabs(x);
    if (x > a) return 0.0f;
    if (x < 0.0001f) return 1.0f;
    float pi_x = pi*x;
    return a*(sin(pi_x)*(sin((pi_x)/a))/(pi_x*pi_x));
  }

  // Blackman-Harris interpolation
  float blackman_harris(float x) {
    if (x < -1.0f || x > 1.0f) return 0.0f;
    x /= 1.5f;
    x = (x + 1.0f) * 0.5f;
    const float a0   = 0.35875f;
    const float a1   = -0.48829f;
    const float a2   = 0.14128f;
    const float a3   = 0.01168f;
    float cos2pix = cos(2.0f * pi * x);
    float cos4pix = 2.0f * cos2pix * cos2pix - 1.0f;
    float cos6pix = cos2pix * (2.0f * cos4pix - 1.0f);
    return a0 + a1 * cos2pix + a2 * cos4pix - a3 * cos6pix;
  }

  float lerp(float a, float b, float f) { // Linear interpolation between a and b given position f
    return a + f * (b - a);
  }
 
  float get_weight(float x) {
    // Calculate linear interpolation at float position x in lookup
    float d = fabs(x)/n*wlut_size;
    int d0 = floor(d);
    int d1 = ceil(d);
    return lerp(weights[d0], weights[d1], d-d0);
  }

  // Choose filter interpolation to weight value x
  float weight(float x) {
    if (filter == 0) { // Blackman-Harris
      return blackman_harris(x);
    } else if (filter == 1 || filter == 2) { // Lanczos
      return lanczos(x, n);
    } else if (filter == 3) { // Cubic
      return bicubic(x, 0.0f, 0.0f, n);
    } else if (filter == 4) { // Mitchell
      return bicubic(x, 0.33333333f, 0.33333333f, n);
    } else if (filter == 5) { // Keys or Catmull-Rom
      return catrom(x);
    } else if (filter == 6) { // Simon
      return bicubic(x, 0.75f, 0.0f, n);
    } else if (filter == 7) { // Rifman
      return bicubic(x, 1.0f, 0.0f, n);
    } else if (filter == 8) { // Parzen
      return bicubic(x, 0.0f, 1.0f, n);
    } else if (filter == 9) { // Gaussian
      return gaussian(x, 6);
    } else if (filter == 10) { // Sharp Gaussian
      return gaussian(x, 4.25f);
    } else { // Default
      return blackman_harris(x);
    }
  }
 
 
  // Sample pixel at continuous float position (x, y) in channel k
  float sample(float x, float y, int k) {
    int u0 = round(x); // nearest x
    int v0 = round(y); // nearest y

    // Normalization factor for lanczos & blackman-harris & gaussian filter windows
    // These weighting functions do not sum to 1 and cause variations in constant input values
    // We sum the weights and then normalize the value after to counteract this
    bool normalize = (filter < 3 || filter == 9 || filter == 10);
    float norm = 0.0f;

    // Loop over neighboring pixels and return weighted sum of pixel values
    float q = 0.0f;
    for (int j = -n; j <= n; j++) {
      int v = v0 + j;
      float p = 0.0f;
      float row_norm = 0.0f;
      for (int i = -n; i <= n; i++) {
        int u = u0 + i;
        float c = src(u, v, k);
        // float w = weight(u-x);
        float w = get_weight(u-x);
        if (rcomp) c = log2shaper(c, 0);
        p += c * w;
        if (normalize) row_norm += w;
      }
      // float w = weight(v-y);
      float w = get_weight(v-y);
      q += p * w;
      if (normalize) norm += row_norm * w;
    }
    if (normalize) q /= norm;
    if (rcomp) return log2shaper(q, 1);
    return q;
  }
 
  void process(int2 pos) {
    // convert from pixel coords to NDC for mathmap code
    float x = (pos.x / width) * 2.0f - 1.0f;
    float y = (pos.y / height) * 2.0f - 1.0f;

    float xr = x * Drpp;
    float yr = (y - 2.0f * vsh) * Drpp;
   
    float azi = d * atan2(xr, d);
    float alt = atan2(yr * (eye + cos(azi)), d);
   
    float sx = Sppr * azi;
    float sy = Sppr * alt;
   
    sx = sx + width * pan / 360.0f;
    if (sx > 1.0f) {
      sx = sx - W;
    } else if (sx < -1.0f) {
      sx = sx + W;
    }
   
    sx = ((sx + 1.0f) * 0.5f) * width;
    sy = ((sy + 1.0f) * 0.5f) * height;
   
    float4 out;
   
    out[0] = sample(sx, sy, 0);
    out[1] = sample(sx, sy, 1);
    out[2] = sample(sx, sy, 2);
    out[3] = sample(sx, sy, 3);
   
    dst() = out;
  }
};
