// Dynamic cover colors come from book data — kept as inline style.
// Structural classes use Tailwind.
export function BookCover({ color, title, size = "md", imageUrl }) {
  const dims = { lg: [80, 112], md: [64, 88], sm: [40, 56] };
  const [w, h] = dims[size] || dims.md;

  return (
    <div
      className={`rounded flex-shrink-0 relative overflow-hidden ${imageUrl ? "book-cover-uninvert" : ""}`}
      style={{
        width: w,
        height: h,
        background: imageUrl
          ? `url(${imageUrl}) center/cover no-repeat`
          : `linear-gradient(145deg, ${color}ee, ${color}77)`,
        boxShadow: `2px 3px 12px ${color}44, inset -2px 0 5px rgba(0,0,0,0.35)`,
      }}
    >
      {!imageUrl && (
        <div className="absolute inset-0" style={{ background: "linear-gradient(to right, rgba(255,255,255,0.1) 0%, transparent 35%)" }} />
      )}
      {!imageUrl && (
        <div className="absolute bottom-0 left-0 right-0 p-1">
          <p className="font-mono text-white/50 leading-tight" style={{ fontSize: 5 }}>
            {title?.slice(0, 20)}
          </p>
        </div>
      )}
    </div>
  );
}
