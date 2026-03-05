import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SmartMine — AI Safety Detection with History",
  description:
    "ResNet-50 powered mine safety image classification with database-backed prediction history. Upload an image to receive an instant safe / unsafe prediction, track results over time, and view statistics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
